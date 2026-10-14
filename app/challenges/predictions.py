"""Predict hidden labels. Scored against a held-out answer key with no model call.

This is the shape of an Aletheia-style lie-detection challenge: players get transcripts
or activations for a set of item ids (public), probe or classify them however they like,
and submit one probability per id. The server holds the true labels and reports AUROC
(or balanced accuracy, or accuracy). Because scoring is pure arithmetic on a stored key, it costs nothing and
needs no NDIF budget.

Submission format, one line per item:  <item_id>,<probability of label 1>
Lines starting with # are ignored. Every id in the key must appear exactly once.

Config:
  labels       {id: 0|1} inline (small sets, tests), or
  labels_file  path to a JSON {id: 0|1} kept out of git (data/private/)
  metric       "auroc" (default), "balanced_accuracy", or "accuracy".
               The last two threshold the probability at 0.5. Balanced accuracy is
               the mean of the true positive and true negative rates, the metric
               Aletheia's Quest used for its weekly prizes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.challenges.base import Challenge, Outcome
from app.errors import ValidationError

MAX_ITEMS = 20000
METRICS = ("auroc", "balanced_accuracy", "accuracy")
THRESHOLD = 0.5


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative (ties count half).
    Rank-sum form of the Mann-Whitney U statistic, so it is exact and O(n log n)."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs at least one item of each label")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):          # average ranks over ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray,
                      threshold: float = THRESHOLD) -> float:
    """Mean of the true positive rate and the true negative rate, predicting label 1
    when the score is at least `threshold`. Always guessing one class scores 0.5."""
    pos = labels == 1
    if not pos.any() or pos.all():
        raise ValueError("balanced accuracy needs at least one item of each label")
    pred = scores >= threshold
    tpr = pred[pos].mean()
    tnr = (~pred[~pos]).mean()
    return float((tpr + tnr) / 2.0)


def accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = THRESHOLD) -> float:
    return float(((scores >= threshold).astype(int) == labels).mean())


SCORERS = {"auroc": auroc, "balanced_accuracy": balanced_accuracy, "accuracy": accuracy}


class PredictionsChallenge(Challenge):
    kind = "predictions"
    mode = "score"

    def validate_config(self) -> None:
        if not (self.config.get("labels") or self.config.get("labels_file")):
            raise ValueError("predictions challenge needs config.labels or config.labels_file")
        if self.config.get("metric", "auroc") not in METRICS:
            raise ValueError("config.metric must be one of " + ", ".join(METRICS))

    def _labels(self) -> dict[str, int]:
        if self.config.get("labels"):
            raw = self.config["labels"]
        else:
            raw = json.loads(Path(self.config["labels_file"]).read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in raw.items()}

    def parse(self, answer: str) -> dict[str, float]:
        preds: dict[str, float] = {}
        for n, line in enumerate(answer.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                raise ValidationError(f"Line {n}: expected 'id,probability'.")
            item, value = parts
            try:
                p = float(value)
            except ValueError:
                raise ValidationError(f"Line {n}: {value!r} is not a number.") from None
            if not 0.0 <= p <= 1.0:
                raise ValidationError(f"Line {n}: probability must be between 0 and 1.")
            if item in preds:
                raise ValidationError(f"Line {n}: id {item!r} appears twice.")
            preds[item] = p
            if len(preds) > MAX_ITEMS:
                raise ValidationError(f"Too many lines (max {MAX_ITEMS}).")
        return preds

    def answer_key(self, answer: str) -> str:
        # Same predictions in a different line order are the same answer.
        preds = self.parse(answer)
        return json.dumps(sorted(preds.items()))

    def evaluate(self, answer: str) -> Outcome:
        preds = self.parse(answer)
        labels = self._labels()
        missing = [k for k in labels if k not in preds]
        extra = [k for k in preds if k not in labels]
        if missing or extra:
            raise ValidationError(
                f"Predictions must cover exactly the {len(labels)} ids in the item set "
                f"({len(missing)} missing, {len(extra)} unknown).")
        ids = sorted(labels)
        y = np.array([labels[i] for i in ids])
        s = np.array([preds[i] for i in ids], dtype=np.float64)
        score = SCORERS[self.config.get("metric", "auroc")](y, s)
        return Outcome(correct=None, score=score, detail={"n": len(ids)})

    def public_config(self) -> dict:
        return {"metric": self.config.get("metric", "auroc"),
                "item_ids": self.config.get("public_item_ids_url", "")}
