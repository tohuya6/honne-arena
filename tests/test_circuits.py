"""The circuits kind: answer parsing, the IOI prompts, and faithfulness.

Fast tests use a tiny random GPT-2 (tests/test_local_model.py) for the mechanics. The
slow ones load real GPT-2 small and check the numbers that make the challenge worth
playing: the full model does IOI, and the known circuit beats random head sets."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pytest

from app import challenges
from app.challenges import circuits
from app.challenges.circuits import CircuitsChallenge, canonical, parse_heads
from app.config import Settings
from app.errors import ScoringUnavailable, ValidationError

ROOT = Path(__file__).resolve().parent.parent


def _row(cid=7001, **cfg):
    return {"id": cid, "slug": f"circ-{cid}", "title": "Circuits", "kind": "circuits",
            "config": cfg}


def _never_remote():
    raise AssertionError("circuits must never make a remote call")


@pytest.fixture(autouse=True)
def _clean():
    circuits.reset_scorers()
    yield
    circuits.reset_scorers()


# ── Answers ──────────────────────────────────────────────────────────────────

def test_parse_accepts_commas_spaces_and_newlines():
    assert parse_heads("9.9, 9.6\n10.0  0.1,\t3.0") == [(0, 1), (3, 0), (9, 6), (9, 9), (10, 0)]
    assert parse_heads(" ,9.9,, ") == [(9, 9)]


@pytest.mark.parametrize("bad, why", [
    ("9.12", "head must be 0-11"),
    ("12.0", "layer must be 0-11"),
    ("9.9 9.9", "listed twice"),
    ("9.9, 09.9", "listed twice"),
    ("a.b", "is not a head"),
    ("9", "is not a head"),
    ("9.9.1", "is not a head"),
    ("-1.0", "is not a head"),
    ("9.9; 9.6", "is not a head"),
])
def test_parse_rejects_bad_heads(bad, why):
    with pytest.raises(ValidationError, match=why):
        parse_heads(bad)


def test_answer_key_is_canonical_and_budget_is_enforced():
    ch = CircuitsChallenge(_row(max_heads=3))
    assert ch.answer_key("10.0\n9.9, 9.6") == "9.6,9.9,10.0"
    assert ch.answer_key("9.6 9.9 10.0") == ch.answer_key("10.0,9.9,9.6")
    with pytest.raises(ValidationError, match="budget is 3"):
        ch.answer_key("1.1 2.2 3.3 4.4")
    with pytest.raises(ValidationError, match="at least one head"):
        ch.answer_key(" , ")
    assert canonical([(10, 0), (9, 9)]) == "9.9,10.0"


def test_config_validation_and_registration():
    assert challenges.KINDS["circuits"] is CircuitsChallenge
    ch = challenges.load(_row())
    assert (ch.mode, ch.compute_mode(), ch.uses_ndif) == ("score", "local", False)
    assert ch.config["model_id"] == "gpt2" and ch.config["split"] == "public"
    assert ch.higher_is_better
    with pytest.raises(ValueError, match="seed_file"):
        CircuitsChallenge(_row(split="hidden"))
    with pytest.raises(ValueError, match="split"):
        CircuitsChallenge(_row(split="secret"))
    with pytest.raises(ValueError, match="unknown model"):
        CircuitsChallenge(_row(model_id="some/other-model"))
    with pytest.raises(ValueError, match="max_heads"):
        CircuitsChallenge(_row(max_heads=0))
    with pytest.raises(ValueError, match="metric"):
        CircuitsChallenge(_row(metric="auc"))
    assert not CircuitsChallenge(_row(metric="distance")).higher_is_better


def test_public_config_never_shows_the_seed_file():
    ch = CircuitsChallenge(_row(split="hidden", seed_file="data/private/circuits/x.json",
                                max_heads=20))
    pub = ch.public_config()
    assert "seed_file" not in json.dumps(pub)
    assert pub["max_heads"] == 20 and pub["split"] == "hidden" and pub["n_prompts"] == 180


# ── Prompts ──────────────────────────────────────────────────────────────────

def test_ioi_prompts_are_deterministic_balanced_and_well_formed():
    a = circuits.generate_ioi(123, 4)
    assert a == circuits.generate_ioi(123, 4)
    assert a != circuits.generate_ioi(124, 4)
    assert len(a) == len(circuits.TEMPLATES) * 2 * 4
    assert sum(p.order == "ABBA" for p in a) == sum(p.order == "BABA" for p in a)
    for p in a:
        words = re.findall(r"\w+", p.text)
        # IO is named once, S twice, and S is the last name before the answer.
        assert p.io != p.s and words.count(p.io) == 1 and words.count(p.s) == 2
        assert [w for w in words if w in (p.io, p.s)][-1] == p.s
        first = words[:words.index(p.io) + 1]
        assert (p.order == "ABBA") == (p.s not in first)


def test_abc_prompts_name_three_different_people():
    for p in circuits.generate_abc(5, 3):
        names = [w.strip(".,") for w in p.text.split() if w.strip(".,") in circuits.NAMES]
        assert len(names) == 3 and len(set(names)) == 3


def test_hidden_split_skips_every_public_prompt(tmp_path):
    seed_file = tmp_path / "seed.json"
    # A hidden seed equal to the public one would regenerate the public prompts, so
    # this is the worst case for the exclusion.
    seed_file.write_text(json.dumps({"seed": circuits.PUBLIC_SEED}))
    ch = CircuitsChallenge(_row(split="hidden", seed_file=str(seed_file)))
    hidden, _abc, _seed = ch.prompts()
    public = {p.text for p in circuits.public_prompts()}
    assert len(hidden) == 180 and not public & {p.text for p in hidden}


def test_hidden_split_without_its_seed_is_unavailable(tmp_path):
    ch = CircuitsChallenge(_row(split="hidden", seed_file=str(tmp_path / "missing.json")))
    with pytest.raises(ScoringUnavailable):
        ch.prepare(Settings(_env_file=None, cache_dir=str(tmp_path)), _never_remote)


def test_published_public_prompts_match_the_generator():
    path = ROOT / "data/public/circuits/ioi_public.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["seed"] == circuits.PUBLIC_SEED
    assert data["templates"] == list(circuits.TEMPLATES)
    assert data["prompts"] == [p._asdict() for p in circuits.public_prompts()]
    assert data["abc_prompts"] == [p._asdict() for p in
                                   circuits.generate_abc(circuits.PUBLIC_SEED + 1, 32)]


# ── Faithfulness on a tiny random GPT-2 ──────────────────────────────────────

TINY = {"model_id": "tiny-gpt2-circuits", "n_layers": 2, "n_heads": 4,
        "n_per_template": 2, "n_mean_per_template": 4, "max_heads": 8}


@pytest.fixture
def tiny(tmp_path):
    """A tiny model for which LD_full > LD_ablated on the prompts (a random model may
    not do IOI at all, and then the challenge refuses to score)."""
    pytest.importorskip("torch")
    from app import local_model
    from tests.test_local_model import tiny_lm

    settings = Settings(_env_file=None, cache_dir=str(tmp_path / "cache"))
    for seed in range(50):
        tiny_lm(TINY["model_id"], seed=seed)
        ch = CircuitsChallenge(_row(**TINY))
        try:
            ch.prepare(settings, _never_remote)
        except ScoringUnavailable:
            circuits.reset_scorers()
            continue
        yield ch, settings
        local_model.clear()
        return
    pytest.fail("no tiny model seed gave LD_full > LD_ablated")


ALL_TINY = "0.0 0.1 0.2 0.3 1.0 1.1 1.2 1.3"


def test_all_heads_score_one_and_none_score_zero(tiny):
    ch, _settings = tiny
    out = ch.evaluate(ALL_TINY)
    assert out.score == pytest.approx(1.0, abs=1e-6)
    assert out.detail["heads"] == 8
    scorer = circuits._SCORERS[ch.id][1]
    f_empty, _ld = scorer.faithfulness([])
    assert f_empty == pytest.approx(0.0, abs=1e-6)
    some = ch.evaluate("1.2, 0.3").score
    assert np.isfinite(some) and some != pytest.approx(1.0)


def test_distance_metric_is_one_minus_faithfulness(tiny):
    ch, _settings = tiny
    faith = ch.evaluate("1.2, 0.3").score
    dist = CircuitsChallenge(_row(**TINY, metric="distance"))
    dist.prepare(_settings, _never_remote)
    assert dist.evaluate("1.2, 0.3").score == pytest.approx(abs(1 - faith), abs=1e-6)


def test_baseline_is_cached_on_disk_and_reused(tiny, monkeypatch):
    ch, settings = tiny
    files = list(Path(settings.cache_dir).glob(f"circuits_{ch.slug}_*.npz"))
    assert len(files) == 1
    first = ch.evaluate("1.2, 0.3").score
    circuits.reset_scorers()

    def boom(*a, **k):
        raise AssertionError("means should come from the cache")
    monkeypatch.setattr(circuits.CircuitScorer, "_means", boom)
    ch.prepare(settings, _never_remote)
    assert ch.evaluate("1.2, 0.3").score == pytest.approx(first, abs=1e-6)


def test_prepare_is_cheap_after_the_first_call(tiny, monkeypatch):
    ch, settings = tiny
    monkeypatch.setattr(circuits, "CircuitScorer", None)   # building again would crash
    ch.prepare(settings, _never_remote)


# ── Real GPT-2 small ─────────────────────────────────────────────────────────

IOI_CIRCUIT_23 = ("9.9 9.6 10.0 10.10 10.6 10.2 10.1 11.2 9.7 9.0 11.9 10.7 11.10 "
                  "7.3 7.9 8.6 8.10 5.5 6.9 0.1 3.0 2.2 4.11")
IOI_CIRCUIT_26 = IOI_CIRCUIT_23 + " 0.10 5.8 5.9"


@pytest.fixture(scope="module")
def gpt2(tmp_path_factory):
    pytest.importorskip("torch")
    from tests.test_local_model import require_cached_gpt2

    require_cached_gpt2()
    settings = Settings(_env_file=None, cache_dir=str(tmp_path_factory.mktemp("cache")))
    ch = CircuitsChallenge(_row(7900, max_heads=144))
    ch.prepare(settings, _never_remote)

    def ready():
        # The autouse reset empties the scorer cache between tests. The baseline is on
        # disk by then, so this only rebuilds the prompt tensors.
        ch.prepare(settings, _never_remote)
        return ch
    return ready


@pytest.mark.slow
def test_gpt2_does_ioi_and_all_heads_are_faithful(gpt2):
    gpt2 = gpt2()
    scorer = circuits._SCORERS[gpt2.id][1]
    assert scorer.ld_full > 2.5
    assert abs(scorer.ld_ablated) < 0.5
    every = " ".join(f"{l}.{h}" for l in range(12) for h in range(12))
    assert gpt2.evaluate(every).score == pytest.approx(1.0, abs=1e-4)


@pytest.mark.slow
def test_known_circuit_beats_random_head_sets(gpt2):
    gpt2 = gpt2()
    f23 = gpt2.evaluate(IOI_CIRCUIT_23).score
    f26 = gpt2.evaluate(IOI_CIRCUIT_26).score
    assert f23 > 0.6 and f26 > 0.8
    rng = random.Random(0)
    heads = [f"{l}.{h}" for l in range(12) for h in range(12)]
    rand = [gpt2.evaluate(" ".join(rng.sample(heads, 23))).score for _ in range(10)]
    assert max(rand) < f23 - 0.3


def test_seeds_load_and_keep_the_hidden_seed_out_of_the_repo():
    seeds = json.loads((ROOT / "seeds/circuits.json").read_text(encoding="utf-8"))
    by_tier = {s["tier"]: CircuitsChallenge({"id": 0, **s}) for s in seeds}
    assert by_tier["learn"].config["split"] == "public"
    hidden = by_tier["compete"]
    assert hidden.config["split"] == "hidden"
    assert hidden.config["seed_file"].startswith("data/private/")
    assert hidden.max_heads < by_tier["learn"].max_heads
