"""Baselines for the lie-detection track, scored the way the arena scores players.

Every baseline writes a submission (id,probability lines) for the test split and runs
it through PredictionsChallenge.evaluate with the private labels, under each metric.

  perfect     the labels themselves (checks the pipeline, should be 1.0)
  random      uniform random probabilities (should be about 0.5)
  white-box   logistic probe on residual-stream activations, trained on the public
              train split. Cities: GPT-2 small, last token. Liar: Qwen2.5-0.5B-Instruct,
              mean over the answer tokens. The layer is picked by cross-validation on
              the train split only.
  black-box   liar only. A TF-IDF logistic classifier on the answer text, and an
              LLM-as-judge baseline that asks the same Qwen model whether the answer is
              correct and uses P("No") as the lie score.

Needs data/private/lie/ (labels), so only organizers can run it. From the repo root:
  HF_HOME=... python scripts/make_lie_baselines.py [cities|liar ...]

Writes submissions to data/private/lie/baselines/ and scores (no labels) to
data/public/lie/<name>/baselines.json.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.challenges.predictions import METRICS, PredictionsChallenge  # noqa: E402
from make_lie_common import (GPT2, GPT2_REVISION, PRIVATE, PUBLIC, QWEN,  # noqa: E402
                             QWEN_REVISION, read_jsonl)

THREADS = 16
OUT = PRIVATE / "baselines"


# ── Scoring ──

def score(name: str, labels_file: Path, preds: dict[str, float]) -> dict[str, float]:
    text = "\n".join(f"{k},{v:.6f}" for k, v in sorted(preds.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.csv").write_text(text + "\n", encoding="utf-8")
    result = {}
    for metric in METRICS:
        row = {"id": 0, "slug": name, "title": name, "kind": "predictions",
               "config": {"labels_file": str(labels_file), "metric": metric}}
        result[metric] = round(PredictionsChallenge(row).evaluate(text).score, 4)
    print(f"  {name:32s} " + "  ".join(f"{m} {v:.3f}" for m, v in result.items()))
    return result


def random_preds(ids: list[str], seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {i: float(p) for i, p in zip(ids, rng.random(len(ids)))}


def random_baseline(name: str, labels_file: Path, ids: list[str],
                    draws: int = 200) -> dict:
    """One saved random submission (seed 0), plus the mean and spread over many
    draws, since a single draw on a few hundred items can land well away from 0.5."""
    result = score(name, labels_file, random_preds(ids))
    many = {m: [] for m in METRICS}
    for seed in range(1, draws + 1):
        text = "\n".join(f"{k},{v:.6f}" for k, v in random_preds(ids, seed).items())
        for m in METRICS:
            row = {"id": 0, "slug": name, "title": name, "kind": "predictions",
                   "config": {"labels_file": str(labels_file), "metric": m}}
            many[m].append(PredictionsChallenge(row).evaluate(text).score)
    result[f"mean_over_{draws}_draws"] = {m: round(float(np.mean(v)), 4)
                                          for m, v in many.items()}
    result[f"sd_over_{draws}_draws"] = {m: round(float(np.std(v)), 4)
                                        for m, v in many.items()}
    print(f"  {'':32s} mean over {draws} draws: "
          + "  ".join(f"{m} {v:.3f}" for m, v in result[f'mean_over_{draws}_draws'].items()))
    return result


# ── Logistic regression (numpy, so the base install is enough) ──

def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression by Newton's method. Inputs are standardized
    by the caller. Returns (weights, bias)."""
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0                                   # no penalty on the bias
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-xb @ w))
        grad = xb.T @ (p - y) / n + reg * w
        hess = (xb * (p * (1 - p))[:, None]).T @ xb / n + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w[:-1], float(w[-1])


class Probe:
    def __init__(self, l2: float = 1e-2):
        self.l2 = l2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Probe":
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1e-6
        self.w, self.b = fit_logistic((x - self.mu) / self.sd, y.astype(float), self.l2)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mu) / self.sd) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))


def cv_auroc(x: np.ndarray, y: np.ndarray, l2: float, folds: int = 5) -> float:
    from app.challenges.predictions import auroc
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(y)) % folds
    scores = np.zeros(len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        scores[te] = Probe(l2).fit(x[tr], y[tr]).predict(x[te])
    return auroc(y, scores)


def pick_layer(feats: np.ndarray, y: np.ndarray, l2: float) -> tuple[int, dict]:
    """feats: [layers, n, d]. Picks the layer with the best cross-validated AUROC on
    the train split."""
    cv = {layer: round(cv_auroc(feats[layer], y, l2), 4) for layer in range(len(feats))}
    best = max(cv, key=lambda k: (cv[k], -k))
    return best, cv


# ── Models ──

def load(model_id: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                 dtype=torch.float32)
    model.eval()
    return tok, model


def gpt2_last_token(texts: list[str]) -> np.ndarray:
    """Residual stream at the last token of each statement, every layer.
    Returns [n_layers + 1, n, d] (index 0 is the embeddings)."""
    import torch
    tok, model = load(GPT2, GPT2_REVISION)
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt")
            hs = model(**ids, output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, -1] for h in hs]).numpy())
    return np.stack(out, axis=1)


def qwen_answer_mean(rows: list[dict], tok=None, model=None) -> np.ndarray:
    """Mean residual stream over the answer tokens of each transcript, every layer.
    The transcript is the question as the user turn and the answer as the assistant
    turn, with Qwen's default system prompt (what apply_chat_template adds when there
    is no system message). Returns [n_layers + 1, n, d]."""
    import torch
    if tok is None:
        tok, model = load(QWEN, QWEN_REVISION)
    out = []
    with torch.no_grad():
        for r in rows:
            ids, start, end = qwen_transcript(tok, r["question"], r["answer"])
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, start:end].mean(0) for h in hs]).numpy())
    return np.stack(out, axis=1)


def qwen_transcript(tok, question: str, answer: str):
    """Token ids of the chat transcript and the [start, end) span of the answer."""
    import torch
    user = [{"role": "user", "content": question}]
    prefix = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": answer}],
                                   tokenize=False)
    assert full.startswith(prefix + answer)
    start = len(tok(prefix)["input_ids"])
    end = len(tok(prefix + answer)["input_ids"])
    ids = torch.tensor([tok(full)["input_ids"]])
    return ids, start, end


def qwen_judge(rows: list[dict], tok, model) -> np.ndarray:
    """P("No") after asking the model whether the answer is correct, renormalized over
    "Yes" and "No". Black-box: it only reads the transcript text."""
    import torch
    yes = tok(" Yes")["input_ids"][-1], tok("Yes")["input_ids"][-1]
    no = tok(" No")["input_ids"][-1], tok("No")["input_ids"][-1]
    scores = []
    with torch.no_grad():
        for r in rows:
            msg = (f"Question: {r['question']}\nAnswer: {r['answer']}\n\n"
                   "Is this answer factually correct? Reply with Yes or No.")
            text = tok.apply_chat_template([{"role": "user", "content": msg}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt")
            logp = torch.log_softmax(model(**ids).logits[0, -1], -1)
            p_yes = torch.logsumexp(logp[list(yes)], 0)
            p_no = torch.logsumexp(logp[list(no)], 0)
            scores.append(float(torch.sigmoid(p_no - p_yes)))
    return np.array(scores)


# ── TF-IDF (black-box text baseline) ──

def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return words + [a + "_" + b for a, b in zip(words, words[1:])]


class Tfidf:
    def fit(self, texts: list[str]) -> "Tfidf":
        df = Counter(t for text in texts for t in set(tokens(text)))
        self.vocab = {t: i for i, t in enumerate(sorted(t for t, c in df.items() if c >= 2))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in self.vocab])
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        x = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            for t, c in Counter(tokens(text)).items():
                if t in self.vocab:
                    x[i, self.vocab[t]] = c
        x *= self.idf
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x


# ── Datasets ──

def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_cities() -> dict:
    print("cities")
    train = read_csv(PUBLIC / "cities" / "train.csv")
    test = read_csv(PUBLIC / "cities" / "test.csv")
    labels_file = PRIVATE / "cities_test_labels.json"
    labels = json.loads(labels_file.read_text())
    ids = [r["id"] for r in test]
    y = np.array([int(r["label"]) for r in train])

    results = {"perfect": score("cities_perfect", labels_file,
                                {k: float(v) for k, v in labels.items()}),
               "random": random_baseline("cities_random", labels_file, ids)}
    l2 = 1e-1
    feats_tr = gpt2_last_token([r["statement"] for r in train])
    feats_te = gpt2_last_token([r["statement"] for r in test])
    layer, cv = pick_layer(feats_tr, y, l2)
    probe = Probe(l2).fit(feats_tr[layer], y)
    p = probe.predict(feats_te[layer])
    results["whitebox_gpt2_probe"] = score("cities_whitebox_gpt2_probe", labels_file,
                                           dict(zip(ids, p.tolist())))
    results["whitebox_gpt2_probe"].update(
        {"layer": layer, "l2": l2, "train_cv_auroc_by_layer": cv})
    print(f"  probe layer {layer} (train CV AUROC {cv[layer]})")
    return results


def run_liar() -> dict:
    print("liar")
    train = read_jsonl(PUBLIC / "liar" / "train.jsonl")
    test = read_jsonl(PUBLIC / "liar" / "test.jsonl")
    labels_file = PRIVATE / "liar_test_labels.json"
    labels = json.loads(labels_file.read_text())
    ids = [r["id"] for r in test]
    y = np.array([int(r["label"]) for r in train])

    results = {"perfect": score("liar_perfect", labels_file,
                                {k: float(v) for k, v in labels.items()}),
               "random": random_baseline("liar_random", labels_file, ids)}

    def text(r):
        return r["question"] + " " + r["answer"]
    vec = Tfidf().fit([text(r) for r in train])
    tf = Probe(l2=1e-1).fit(vec.transform([text(r) for r in train]), y)
    p = tf.predict(vec.transform([text(r) for r in test]))
    results["blackbox_tfidf"] = score("liar_blackbox_tfidf", labels_file,
                                      dict(zip(ids, p.tolist())))

    tok, model = load(QWEN, QWEN_REVISION)
    p = qwen_judge(test, tok, model)
    results["blackbox_qwen_judge"] = score("liar_blackbox_qwen_judge", labels_file,
                                           dict(zip(ids, p.tolist())))

    l2 = 1e-1
    feats_tr = qwen_answer_mean(train, tok, model)
    feats_te = qwen_answer_mean(test, tok, model)
    layer, cv = pick_layer(feats_tr, y, l2)
    p = Probe(l2).fit(feats_tr[layer], y).predict(feats_te[layer])
    results["whitebox_qwen_probe"] = score("liar_whitebox_qwen_probe", labels_file,
                                           dict(zip(ids, p.tolist())))
    results["whitebox_qwen_probe"].update(
        {"layer": layer, "l2": l2, "train_cv_auroc_by_layer": cv})
    print(f"  probe layer {layer} (train CV AUROC {cv[layer]})")
    return results


def main() -> None:
    names = sys.argv[1:] or ["cities", "liar"]
    for name in names:
        results = {"cities": run_cities, "liar": run_liar}[name]()
        (PUBLIC / name / "baselines.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
