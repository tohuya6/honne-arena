"""Steering with config.backend = "local": the same shift metric on a small model on
this server, with no NDIF and no call to on_remote_call."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app import scoring
from app.challenges import steering
from app.challenges.steering import SteeringChallenge
from app.config import Settings
from app.errors import ScoringUnavailable, ValidationError

ROOT = Path(__file__).resolve().parent.parent


def _never_remote():
    raise AssertionError("the local backend must never make a remote call")


@pytest.fixture(autouse=True)
def _clean():
    steering.reset_scorers()
    yield
    steering.reset_scorers()


def _row(cid, **cfg):
    return {"id": cid, "slug": f"steer-local-{cid}", "title": "Steer", "kind": "steering",
            "config": cfg}


def test_local_backend_is_a_local_compute_mode():
    ch = SteeringChallenge(_row(8001, backend="local", model_id="gpt2", layer=9,
                                d_file="x.npz", probe_file="y.json"))
    assert ch.compute_mode() == "local" and not ch.uses_ndif


@pytest.fixture
def tiny(tmp_path):
    pytest.importorskip("torch")
    from app import local_model
    from tests.test_local_model import D_MODEL, tiny_lm

    tiny_lm("tiny-gpt2-steer")
    rng = np.random.default_rng(0)
    d = rng.standard_normal(D_MODEL).astype(np.float32)
    np.savez(tmp_path / "d.npz", d=d / np.linalg.norm(d), meta=np.array(json.dumps({})))
    (tmp_path / "probes.json").write_text(json.dumps(
        {"prompts": ["hello world", "the train leaves the station at", "here is a list of"]}))
    cfg = dict(backend="local", model_id="tiny-gpt2-steer", layer=1, token_budget=4,
               d_file=str(tmp_path / "d.npz"), probe_file=str(tmp_path / "probes.json"))
    settings = Settings(_env_file=None, cache_dir=str(tmp_path / "cache"))
    yield cfg, settings, tmp_path
    local_model.clear()


def test_scores_locally_without_remote_calls(tiny):
    cfg, settings, tmp_path = tiny
    ch = SteeringChallenge(_row(8002, **cfg))
    ch.prepare(settings, _never_remote)
    good = ch.evaluate("good great love")
    bad = ch.evaluate("bad terrible hate")
    assert good.detail == {"tokens": 3} and np.isfinite(good.score)
    assert good.score != bad.score
    assert ch.evaluate("good great love").score == good.score       # deterministic
    with pytest.raises(ValidationError, match="budget is 4"):
        ch.evaluate("good good good good good")
    assert len(list((tmp_path / "cache").glob("baseline_*.npy"))) == 1


def test_local_score_matches_the_shift_metric_by_hand(tiny):
    """The scorer is Soham's batched shift over the local read, nothing else."""
    from app import local_model

    cfg, settings, _tmp = tiny
    count, score = steering.build_scorer(SteeringChallenge(_row(8003, **cfg)), settings,
                                         _never_remote)
    lm = local_model.load("tiny-gpt2-steer")
    d, _ = scoring.load_direction(cfg["d_file"])
    probes = scoring.load_probes(cfg["probe_file"])

    def one(text):
        return local_model.batch_last_resids(lm, [text], 1, prepend_bos=True)[0]

    by_hand = scoring.steering_shift_score("good day", probes, one, d)
    assert score("good day") == pytest.approx(by_hand, abs=1e-5)
    assert count("good day") == 2


def test_direction_of_the_wrong_size_is_refused(tiny):
    cfg, settings, tmp_path = tiny
    np.savez(tmp_path / "wide.npz", d=np.ones(7, dtype=np.float32))
    ch = SteeringChallenge(_row(8004, **{**cfg, "d_file": str(tmp_path / "wide.npz")}))
    with pytest.raises(ScoringUnavailable, match="hidden size"):
        ch.prepare(settings, _never_remote)


def test_committed_gpt2_direction_and_probes_are_well_formed():
    d, meta = scoring.load_direction(ROOT / "data/directions/d_gpt2_L9_sentiment.npz")
    assert d.shape == (768,) and abs(float(np.linalg.norm(d)) - 1) < 1e-5
    assert meta["model_id"] == "gpt2" and meta["layer"] == 9 and meta["prepend_bos"] is True
    probes = scoring.load_probes(ROOT / "data/probes/gpt2_neutral.json")
    assert len(probes) >= 8 and len(set(probes)) == len(probes)


def test_seed_file_points_at_the_committed_files():
    seeds = json.loads((ROOT / "seeds/steering.json").read_text(encoding="utf-8"))
    for s in seeds:
        ch = SteeringChallenge({"id": 0, **s})
        assert (ROOT / ch.config["d_file"]).exists() and (ROOT / ch.config["probe_file"]).exists()
        if s["tier"] == "learn":
            assert ch.compute_mode() == "local"
        else:
            assert ch.compute_mode() == "ndif" and s["hidden"] is True


# ── Real GPT-2 small ─────────────────────────────────────────────────────────

ALIGNED = ["I love it, wonderful!", "What a beautiful, happy day.",
           "Excellent, fantastic, amazing.", "This is great news and I am so glad."]
OPPOSITE = ["I hate it, terrible!", "What an ugly, miserable day.",
            "Horrible, awful, disgusting.", "This is terrible news and I am so upset."]


@pytest.mark.slow
def test_gpt2_sentiment_prefixes_move_the_score_the_right_way(tmp_path):
    pytest.importorskip("torch")
    from tests.test_local_model import require_cached_gpt2

    require_cached_gpt2()
    seed = next(s for s in json.loads((ROOT / "seeds/steering.json").read_text())
                if s["tier"] == "learn")
    ch = SteeringChallenge({"id": 8100, **seed})
    ch.prepare(Settings(_env_file=None, cache_dir=str(tmp_path)), _never_remote)
    pos = [ch.evaluate(t).score for t in ALIGNED]
    neg = [ch.evaluate(t).score for t in OPPOSITE]
    assert min(pos) > 0 > max(neg)
    assert np.mean(pos) - np.mean(neg) > 0.03
