"""Lie-detection track: the balanced accuracy metric and the dataset invariants.

The private label files are gitignored, so on a fresh clone the tests that need them
are skipped. Run scripts/make_lie_cities.py (and the liar generator) to build them.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from app.challenges.predictions import PredictionsChallenge, balanced_accuracy
from app.errors import ValidationError

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "data" / "public" / "lie"
PRIVATE = ROOT / "data" / "private" / "lie"
SEEDS = ROOT / "seeds" / "lie-detection.json"


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


# ── Datasets ──

def _csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


DATASETS = {
    # name: (train rows, test rows, private labels file, text field)
    "cities": (lambda: _csv(PUBLIC / "cities" / "train.csv"),
               lambda: _csv(PUBLIC / "cities" / "test.csv"),
               PRIVATE / "cities_test_labels.json", "statement"),
    "liar": (lambda: _jsonl(PUBLIC / "liar" / "train.jsonl"),
             lambda: _jsonl(PUBLIC / "liar" / "test.jsonl"),
             PRIVATE / "liar_test_labels.json", "answer"),
}
NAMES = sorted(DATASETS)


def _labels(name):
    path = DATASETS[name][2]
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} is private and not built here")
    return {k: int(v) for k, v in json.loads(path.read_text()).items()}


@pytest.mark.parametrize("name", NAMES)
def test_ids_unique_and_splits_disjoint(name):
    train, test = DATASETS[name][0](), DATASETS[name][1]()
    train_ids = [r["id"] for r in train]
    test_ids = [r["id"] for r in test]
    assert len(set(train_ids)) == len(train_ids)
    assert len(set(test_ids)) == len(test_ids)
    assert not set(train_ids) & set(test_ids)


@pytest.mark.parametrize("name", NAMES)
def test_no_text_in_both_splits(name):
    field = DATASETS[name][3]
    train, test = DATASETS[name][0](), DATASETS[name][1]()
    key = "statement" if name == "cities" else "question"
    assert not {r[key] for r in train} & {r[key] for r in test}
    assert all(r[field].strip() for r in train + test)


def test_cities_never_share_a_city_across_splits():
    def city(r):
        return re.match(r"The city of (.+) is in ", r["statement"]).group(1)
    train, test = DATASETS["cities"][0](), DATASETS["cities"][1]()
    assert not {city(r) for r in train} & {city(r) for r in test}


@pytest.mark.parametrize("name", NAMES)
def test_public_test_files_have_no_labels(name):
    test = DATASETS[name][1]()
    for r in test:
        assert set(r) <= {"id", "statement", "category", "question", "answer"}, set(r)


@pytest.mark.parametrize("name", NAMES)
def test_train_split_is_balanced(name):
    labels = [int(r["label"]) for r in DATASETS[name][0]()]
    assert set(labels) == {0, 1}
    assert sum(labels) * 2 == len(labels)


def test_liar_categories_balanced_in_train():
    rows = DATASETS["liar"][0]()
    for cat in {r["category"] for r in rows}:
        labels = [r["label"] for r in rows if r["category"] == cat]
        assert sum(labels) * 2 == len(labels), cat


@pytest.mark.parametrize("name", NAMES)
def test_every_test_id_has_a_private_label(name):
    labels = _labels(name)
    test_ids = {r["id"] for r in DATASETS[name][1]()}
    assert set(labels) == test_ids
    assert set(labels.values()) == {0, 1}
    assert sum(labels.values()) * 2 == len(labels)        # balanced


def test_liar_test_categories_balanced():
    labels = _labels("liar")
    rows = DATASETS["liar"][1]()
    for cat in {r["category"] for r in rows}:
        y = [labels[r["id"]] for r in rows if r["category"] == cat]
        assert sum(y) * 2 == len(y), cat


@pytest.mark.parametrize("name", NAMES)
def test_perfect_and_inverted_submissions(name):
    labels = _labels(name)
    row = {"id": 1, "slug": name, "title": name, "kind": "predictions",
           "config": {"labels_file": str(DATASETS[name][2]), "metric": "auroc"}}
    c = PredictionsChallenge(row)
    assert c.evaluate(_lines(labels)).score == 1.0
    assert c.evaluate(_lines({k: 1 - v for k, v in labels.items()})).score == 0.0


def test_labels_are_not_tracked_by_git():
    ignore = (ROOT / ".gitignore").read_text()
    assert "data/private/" in ignore.splitlines()


# ── Seeds ──

def test_seed_challenges_load():
    seeds = json.loads(SEEDS.read_text())
    assert {s["tier"] for s in seeds} == {"learn", "compete"}
    for s in seeds:
        assert s["kind"] == "predictions" and s["track"] == "lie-detection"
        assert s["config"]["labels_file"].startswith("data/private/lie/")
        assert "/datasets/lie/" in s["description"]
        assert "labels" not in s["config"]
        url = s["config"]["public_item_ids_url"]
        assert (ROOT / "data" / "public" / url.removeprefix("/datasets/")).exists()
        PredictionsChallenge({"id": 1, **s})          # config validates
