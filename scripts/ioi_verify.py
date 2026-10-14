"""Check the circuits scorer on real GPT-2 small: baselines, the known IOI circuit
against random head sets, and timings.

    HF_HOME=... python -m scripts.ioi_verify [--split public] [--random 20]

Prints a table. Nothing is written except the baseline cache in --cache-dir.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from app.challenges.circuits import CircuitsChallenge, _SCORERS
from app.config import Settings

# Wang et al. 2022, Figure 2 and Section 3.
IOI_CIRCUIT = {
    "name movers": ["9.9", "9.6", "10.0"],
    "backup name movers": ["10.10", "10.6", "10.2", "10.1", "11.2", "9.7", "9.0", "11.9"],
    "negative name movers": ["10.7", "11.10"],
    "S-inhibition": ["7.3", "7.9", "8.6", "8.10"],
    "induction": ["5.5", "6.9"],
    "duplicate token": ["0.1", "3.0"],
    "previous token": ["2.2", "4.11"],
}
# The paper's full 26-head circuit adds one duplicate token head and two induction heads.
PAPER_EXTRA = ["0.10", "5.8", "5.9"]


def _no_remote() -> None:
    raise RuntimeError("the circuits scorer must never make a remote call")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="public", choices=["public", "hidden"])
    ap.add_argument("--seed-file", default="data/private/circuits/ioi-compete.json")
    ap.add_argument("--random", type=int, default=20, help="random head sets per size")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args(argv)

    settings = Settings(_env_file=None, cache_dir=args.cache_dir)
    cfg = {"model_id": "gpt2", "split": args.split, "max_heads": 144}
    if args.split == "hidden":
        cfg["seed_file"] = args.seed_file
    ch = CircuitsChallenge({"id": -1, "slug": f"verify-{args.split}", "title": "verify",
                            "config": cfg})
    t0 = time.perf_counter()
    ch.prepare(settings, _no_remote)
    t_prep = time.perf_counter() - t0
    scorer = _SCORERS[ch.id][1]
    per = scorer.per_prompt_ld(np.zeros((12, 12), dtype=bool))
    print(f"split={args.split} prompts={len(scorer.prompts)} prepare={t_prep:.2f}s")
    print(f"LD_full={scorer.ld_full:.4f} (per-prompt min {per.min():.3f}, "
          f"frac>0 {np.mean(per > 0):.3f})  LD_ablated={scorer.ld_ablated:.4f}")

    def run(heads):
        t = time.perf_counter()
        out = ch.evaluate(", ".join(heads))
        return out.score, time.perf_counter() - t

    rows = []
    all_heads = [f"{l}.{h}" for l in range(12) for h in range(12)]
    rows.append(("all 144 heads", 144, *run(all_heads)))
    f_empty = (scorer.mean_ld(np.ones((12, 12), dtype=bool)) - scorer.ld_ablated) / (
        scorer.ld_full - scorer.ld_ablated)
    rows.append(("empty set", 0, f_empty, 0.0))
    circuit = [h for group in IOI_CIRCUIT.values() for h in group]
    rows.append(("IOI circuit (Wang et al.)", len(circuit), *run(circuit)))
    rows.append(("IOI circuit, all 26 heads", len(circuit) + 3, *run(circuit + PAPER_EXTRA)))
    no_neg = [h for g, hs in IOI_CIRCUIT.items() if g != "negative name movers" for h in hs]
    rows.append(("IOI circuit minus negative NMs", len(no_neg), *run(no_neg)))
    rows.append(("name movers + S-inh + induction", 9, *run(
        IOI_CIRCUIT["name movers"] + IOI_CIRCUIT["S-inhibition"] + IOI_CIRCUIT["induction"])))
    print(f"{'set':34s} {'heads':>5s} {'faith':>8s} {'time':>7s}")
    for name, n, score, dt in rows:
        print(f"{name:34s} {n:5d} {score:8.4f} {dt:6.2f}s")

    rng = random.Random(0)
    times = []
    for size in (len(circuit), 26, 30):
        scores = []
        for _ in range(args.random):
            s, dt = run(rng.sample(all_heads, size))
            scores.append(s)
            times.append(dt)
        a = np.array(scores)
        print(f"random size {size:3d} (n={len(a)}): mean {a.mean():.4f} sd {a.std():.4f} "
              f"min {a.min():.4f} median {np.median(a):.4f} max {a.max():.4f}")
    print(f"evaluate time: median {np.median(times):.2f}s max {max(times):.2f}s")


if __name__ == "__main__":
    main()
