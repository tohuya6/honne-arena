"""Lie-detection track: the balanced accuracy metric and the dataset invariants.

The private label files are gitignored, so on a fresh clone the tests that need them
are skipped. Run scripts/make_lie_cities.py (and the liar generator) to build them.
"""

import numpy as np
import pytest

from app.challenges.predictions import PredictionsChallenge, balanced_accuracy
from app.errors import ValidationError


def _challenge(labels, metric):
    return PredictionsChallenge({"id": 1, "slug": "x", "title": "x", "kind": "predictions",
                                 "config": {"labels": labels, "metric": metric}})


def _lines(preds):
    return "\n".join(f"{k},{v}" for k, v in preds.items())


# ── Metric ──

def test_balanced_accuracy_matches_rate_definition():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.2).astype(int)          # imbalanced on purpose
    s = rng.random(300)
    pred = s >= 0.5
    tpr = (pred & (y == 1)).sum() / (y == 1).sum()
    tnr = (~pred & (y == 0)).sum() / (y == 0).sum()
    assert balanced_accuracy(y, s) == pytest.approx((tpr + tnr) / 2)


def test_balanced_accuracy_constant_guess_is_half():
    y = np.array([1, 0, 0, 0, 0, 0])
    assert balanced_accuracy(y, np.ones(6)) == 0.5
    assert balanced_accuracy(y, np.zeros(6)) == 0.5


def test_balanced_accuracy_threshold_is_inclusive():
    y = np.array([1, 0])
    assert balanced_accuracy(y, np.array([0.5, 0.49])) == 1.0


def test_balanced_accuracy_needs_both_labels():
    with pytest.raises(ValueError):
        balanced_accuracy(np.array([1, 1]), np.array([0.2, 0.9]))


def test_metrics_through_evaluate():
    labels = {"a": 1, "b": 0, "c": 0, "d": 0}
    preds = {"a": 0.9, "b": 0.6, "c": 0.1, "d": 0.2}
    assert _challenge(labels, "balanced_accuracy").evaluate(_lines(preds)).score \
        == pytest.approx((1.0 + 2 / 3) / 2)
    assert _challenge(labels, "accuracy").evaluate(_lines(preds)).score == 0.75
    assert _challenge(labels, "auroc").evaluate(_lines(preds)).score == 1.0


def test_default_metric_is_still_auroc():
    c = PredictionsChallenge({"id": 1, "slug": "x", "title": "x", "kind": "predictions",
                              "config": {"labels": {"a": 1, "b": 0}}})
    assert c.public_config()["metric"] == "auroc"
    assert c.evaluate("a,0.2\nb,0.1").score == 1.0


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError):
        _challenge({"a": 1, "b": 0}, "f1")


def test_missing_ids_still_rejected_under_balanced_accuracy():
    with pytest.raises(ValidationError):
        _challenge({"a": 1, "b": 0}, "balanced_accuracy").evaluate("a,0.9")
