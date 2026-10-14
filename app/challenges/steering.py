"""Activation steering, ported from Steering Arena (Soham Padia).

A player submits a short string. The server prepends it to fixed neutral probe prompts,
reads the model's last-token residual at one layer, and scores how far the string moves
that residual along a direction `d`, relative to the probes alone (Steering Arena's
"steering shift"). Higher pushes harder toward the direction.

Every evaluation is ONE batched remote forward over all probes. The probe baseline is
also one forward, computed once per challenge and cached to disk so a restart does not
spend it again.

Config:
  model_id     NDIF-hosted model id, e.g. "meta-llama/Llama-3.1-8B"
  layer        residual layer to read
  d_file       .npz with key "d" (unit direction, hidden-size long)
  probe_file   JSON list (or {"prompts": [...]}) of neutral prompts
  token_budget max tokens in a submission (default 64)
  higher_is_better  default true. Set false for an "anti" board.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from app import scoring
from app.challenges.base import Challenge, Outcome
from app.errors import ScoringUnavailable, ValidationError

# challenge id -> (count_tokens, score_fn). Built lazily on first use.
_SCORERS: dict[int, tuple[Callable[[str], int], Callable[[str], float]]] = {}
# challenge id -> the remote-call hook of the submission being judged right now.
# Submissions are judged one at a time behind the NDIF queue, so one slot is enough.
_HOOKS: dict[int, Callable[[], None]] = {}


def build_scorer(challenge: "SteeringChallenge", settings, on_remote_call: Callable[[], None]):
    """Real scorer over NDIF (or a local model when SCORING_BACKEND=local).

    `on_remote_call` is invoked once per remote forward so the budget ledger sees the
    baseline build too, not just player submissions."""
    from app.ndif_client import ResidualReader

    cfg = challenge.config
    backend = settings.scoring_backend.lower()
    model_id = cfg["model_id"] if backend == "ndif" else settings.local_model_id
    reader = ResidualReader.build(model_id, backend, ndif_key=settings.ndif_api_key,
                                  prepend_bos=settings.prepend_bos)
    d, _meta = scoring.load_direction(cfg["d_file"])
    hidden = reader.hidden_size
    if hidden is not None and hidden != d.shape[0]:
        raise ScoringUnavailable(f"Direction length {d.shape[0]} does not match the model's "
                                 f"hidden size {hidden}.")
    probes = scoring.load_probes(cfg["probe_file"])
    layer = int(cfg["layer"])
    remote = backend == "ndif"

    def batch_fn(texts):
        if remote:
            on_remote_call()
        return reader.batch_last_resids(list(texts), layer)

    base_units = _cached_baseline(challenge, settings, model_id, probes,
                                  lambda: scoring.baseline_unit_rows(probes, batch_fn))

    def count_tokens(text: str) -> int:
        return len(reader.tokenizer(text, add_special_tokens=False)["input_ids"])

    def score_fn(seq: str) -> float:
        shift, _z = scoring.shift_and_specificity(seq, probes, batch_fn, base_units, d)
        return float(shift)

    return count_tokens, score_fn


def _cached_baseline(challenge, settings, model_id, probes, compute):
    joined = "\n".join(probes)
    key = hashlib.sha256(
        f"{model_id}|{challenge.config['layer']}|{challenge.config['d_file']}|{joined}"
        .encode("utf-8")).hexdigest()[:16]
    path = Path(settings.cache_dir) / f"baseline_{challenge.slug}_{key}.npy"
    if path.exists():
        return np.load(path)
    base = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, base)
    return base


class SteeringChallenge(Challenge):
    kind = "steering"
    mode = "score"
    compute = "ndif"

    def compute_mode(self) -> str:
        # config.backend "local" scores on a small model on this server (learn tier,
        # no NDIF spend). Anything else is NDIF.
        return "local" if self.config.get("backend") == "local" else "ndif"

    # Tests swap this for a fake so no test ever builds a real model client.
    scorer_factory = staticmethod(build_scorer)

    def validate_config(self) -> None:
        for k in ("model_id", "layer", "d_file", "probe_file"):
            if k not in self.config:
                raise ValueError(f"steering challenge needs config.{k}")

    @property
    def higher_is_better(self) -> bool:
        return bool(self.config.get("higher_is_better", True))

    def answer_key(self, answer: str) -> str:
        return " ".join(answer.strip().lower().split())

    def prepare(self, settings, on_remote_call) -> None:
        """Point this challenge's remote-call hook at the current submission, building
        the scorer on first use. The scorer is cached across submissions, so it calls
        the hook through _HOOKS rather than capturing the first caller's function."""
        _HOOKS[self.id] = on_remote_call
        if self.id not in _SCORERS:
            cid = self.id
            _SCORERS[cid] = self.scorer_factory(self, settings, lambda: _HOOKS[cid]())

    def evaluate(self, answer: str) -> Outcome:
        count_tokens, score_fn = _SCORERS[self.id]
        n = count_tokens(answer)
        budget = int(self.config.get("token_budget", 64))
        if n > budget:
            raise ValidationError(f"That is {n} tokens. The budget is {budget}.")
        return Outcome(correct=None, score=score_fn(answer), detail={"tokens": n})

    def public_config(self) -> dict:
        return {"model_id": self.config["model_id"], "layer": self.config["layer"],
                "token_budget": int(self.config.get("token_budget", 64)),
                "higher_is_better": self.higher_is_better}


def reset_scorers() -> None:
    _SCORERS.clear()
    _HOOKS.clear()
