"""Build the GPT-2 small sentiment direction for the learn-tier steering challenge.

    HF_HOME=... python -m scripts.make_gpt2_direction [--layer 9] [--sweep] [--check]

The direction is the normalized mean difference of last-token residuals (output of block
`layer`, BOS prepended, the same read the scorer does) between positive and negative
sentences. Each pair differs only in its sentiment words, so length and topic cancel.

The pairs are public on purpose: learn-tier players can rebuild the vector themselves
and see why a prefix scores the way it does.

--sweep   prints held-out separation for every layer (build on even pairs, test on odd)
--check   scores aligned and opposite prefixes with the challenge's own scorer
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODEL_ID = "gpt2"
PROBE_FILE = "data/probes/gpt2_neutral.json"

# (positive, negative). Same words except the sentiment.
PAIRS = [
    ("The movie was wonderful.", "The movie was terrible."),
    ("I love this song.", "I hate this song."),
    ("The food here is delicious.", "The food here is disgusting."),
    ("She had a great day at work.", "She had an awful day at work."),
    ("This is the best book I have read.", "This is the worst book I have read."),
    ("The service was excellent.", "The service was horrible."),
    ("We had a fantastic time on the trip.", "We had a miserable time on the trip."),
    ("He is a kind and generous man.", "He is a cruel and selfish man."),
    ("The weather today is beautiful.", "The weather today is dreadful."),
    ("I am so happy with the results.", "I am so unhappy with the results."),
    ("The new phone works perfectly.", "The new phone works poorly."),
    ("Their wedding was a joyful event.", "Their wedding was a painful event."),
    ("The hotel room was clean and comfortable.", "The hotel room was dirty and uncomfortable."),
    ("Everyone praised the performance.", "Everyone criticized the performance."),
    ("The team played brilliantly.", "The team played badly."),
    ("I feel calm and hopeful.", "I feel anxious and hopeless."),
    ("What a lovely surprise.", "What a nasty surprise."),
    ("The meeting went really well.", "The meeting went really badly."),
    ("My neighbors are friendly.", "My neighbors are hostile."),
    ("The garden looks gorgeous this spring.", "The garden looks ugly this spring."),
    ("I enjoyed every minute of it.", "I hated every minute of it."),
    ("The teacher was patient and helpful.", "The teacher was rude and unhelpful."),
    ("It was a pleasant evening.", "It was an unpleasant evening."),
    ("The results were impressive.", "The results were disappointing."),
    ("Our vacation was perfect.", "Our vacation was a disaster."),
    ("The coffee tastes amazing.", "The coffee tastes awful."),
    ("He gave a brilliant speech.", "He gave a terrible speech."),
    ("The kids were delighted.", "The kids were upset."),
    ("This product is a success.", "This product is a failure."),
    ("I am grateful for my friends.", "I am angry at my friends."),
    ("The city is safe and charming.", "The city is dangerous and grim."),
    ("The ending made me smile.", "The ending made me cry."),
]

# Prefixes for --check. Aligned ones should score above the opposite ones.
CHECK_POSITIVE = [
    "I love it, wonderful!",
    "What a beautiful, happy day.",
    "Excellent, fantastic, amazing.",
    "This is great news and I am so glad.",
]
CHECK_NEGATIVE = [
    "I hate it, terrible!",
    "What an ugly, miserable day.",
    "Horrible, awful, disgusting.",
    "This is terrible news and I am so upset.",
]
CHECK_NEUTRAL = [
    "The table is made of wood.",
    "It is Tuesday.",
    "Page twelve of the report.",
]


def resids(lm, texts, layer):
    from app import local_model

    return local_model.batch_last_resids(lm, texts, layer, prepend_bos=True).astype(np.float64)


def direction(lm, pairs, layer) -> np.ndarray:
    pos = resids(lm, [p for p, _ in pairs], layer)
    neg = resids(lm, [n for _, n in pairs], layer)
    d = pos.mean(axis=0) - neg.mean(axis=0)
    return d / np.linalg.norm(d)


def held_out(lm, layer) -> float:
    """Fraction of held-out pairs whose positive side projects above its negative side,
    for a direction built on the other half."""
    train, test = PAIRS[0::2], PAIRS[1::2]
    d = direction(lm, train, layer)
    pos = resids(lm, [p for p, _ in test], layer) @ d
    neg = resids(lm, [n for _, n in test], layer) @ d
    return float(np.mean(pos > neg))


def check(layer: int, d_file: str) -> None:
    from app.challenges.steering import SteeringChallenge, build_scorer
    from app.config import Settings

    row = {"id": -1, "slug": "check-gpt2-sentiment", "title": "check",
           "config": {"backend": "local", "model_id": MODEL_ID, "layer": layer,
                      "d_file": d_file, "probe_file": PROBE_FILE}}
    ch = SteeringChallenge(row)

    def no_remote():
        raise RuntimeError("the local backend must never make a remote call")

    settings = Settings(_env_file=None, cache_dir="data/cache")
    _count, score = build_scorer(ch, settings, no_remote)
    for label, group in (("aligned", CHECK_POSITIVE), ("opposite", CHECK_NEGATIVE),
                         ("neutral", CHECK_NEUTRAL)):
        vals = [score(s) for s in group]
        for s, v in zip(group, vals):
            print(f"  {label:8s} {v:+.5f}  {s}")
        print(f"{label:8s} mean {np.mean(vals):+.5f}")


def main(argv=None) -> None:
    from app import local_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    lm = local_model.load(MODEL_ID, "cpu")
    if args.sweep:
        for layer in range(lm.n_layers):
            print(f"layer {layer:2d}  held-out separation {held_out(lm, layer):.3f}")
        return

    out = args.out or f"data/directions/d_gpt2_L{args.layer}_sentiment.npz"
    d = direction(lm, PAIRS, args.layer).astype(np.float32)
    meta = {
        "model_id": MODEL_ID, "layer": args.layer, "d_version": f"gpt2_L{args.layer}_sentiment",
        "extraction_method": "mean_difference", "concept": "positive minus negative sentiment",
        "prepend_bos": True, "num_pairs": len(PAIRS),
        "held_out_separation": held_out(lm, args.layer), "backend": "local",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": "scripts/make_gpt2_direction.py. The contrast pairs are in that script.",
        "placeholder": False,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, d=d, meta=np.array(json.dumps(meta)))
    print(f"wrote {out}  held-out separation {meta['held_out_separation']:.3f}")
    if args.check:
        check(args.layer, out)


if __name__ == "__main__":
    main()
