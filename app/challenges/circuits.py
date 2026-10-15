"""Circuit localization on IOI with GPT-2 small, a replica of MIB's circuit track
(Mueller et al., ICML 2025) at the granularity of attention heads.

The task is indirect object identification (Wang et al. 2022). A prompt such as

    "Then, Mary and John went to the store. John gave a ring to"

should be completed with the indirect object (" Mary"), not the subject (" John").
GPT-2 small does this well, and a few dozen of its 144 heads do most of the work.

**Answer.** A set of heads, each written "L.H" (layer, head, both 0-11), separated by
commas, spaces, or newlines. For example `9.9, 9.6, 10.0`. Duplicates are rejected and
the set may hold at most `config.max_heads` heads.

**Score.** Faithfulness of the circuit,

    faithfulness = (LD_circuit - LD_ablated) / (LD_full - LD_ablated)

where LD = logit(IO) - logit(S) at the final position. Each LD is the MEAN over the
challenge's prompts of the per-prompt logit difference, and the ratio is taken of those
means (as in Wang et al. and MIB). The three runs:
- full:     the model as is.
- ablated:  every head mean-ablated.
- circuit:  only the submitted heads keep their real outputs, every other head is
            mean-ablated. MLPs and embeddings are never ablated.
1.0 means the heads alone recover the full model's behaviour, 0 means they recover
none of it. It can go above 1 (a set that leaves out heads that hurt the task, such as
the negative name movers) or below 0.

**Mean-ablation reference.** Position-wise means over an "ABC" distribution built from
the same templates, as in the IOI paper: the second clause names a third person, so the
prompt carries no information about which name is the answer. Every filler (names,
places, objects) is a single GPT-2 token, so all prompts of one template have the same
token length and position t means the same thing in each. The mean for a head is taken
per template and per position, and a prompt is ablated with its own template's means.

**Prompts.** Generated from fixed templates and word lists with a seeded RNG, half in
ABBA order and half in BABA order. `config.split = "public"` uses the public seed, and
its prompts can be published under /datasets/circuits/. `"hidden"` reads its seed from
`config.seed_file` under data/private/ (never committed) and skips any prompt that is
also in the public set, so compete players score on prompts they do not have.

The baseline (LD_full, LD_ablated, and the means) is computed once per challenge and
cached in settings.cache_dir. A submission is then one forward over the prompts.

Config:
  model_id          "gpt2" (GPT-2 small, the default). Other GPT-2 style models need
                    config.n_layers and config.n_heads.
  split             "public" (default) or "hidden"
  seed_file         hidden split only: JSON {"seed": <int>} under data/private/
  n_per_template    IOI prompts per template and order (default 6, so 180 prompts)
  n_mean_per_template  ABC prompts per template for the means (default 32)
  max_heads         head budget (default 30)
  metric            "faithfulness" (default, higher wins, like MIB's CPR) or "distance",
                    |1 - faithfulness|, lower wins, like MIB's CMD. With "faithfulness" a
                    set that leaves out the heads that hurt the task can beat 1.
  prompts_url       optional public link to the prompt file, shown to players
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from pathlib import Path
from typing import NamedTuple

import numpy as np

from app.challenges.base import Challenge, Outcome
from app.errors import ScoringUnavailable, ValidationError

# Bump when the dataset, the templates, or the cached quantities change meaning.
CACHE_VERSION = 1
PUBLIC_SEED = 20221101   # Wang et al. posted the IOI paper on 2022-11-01

# Fillers. Every one is a single GPT-2 token with its leading space, checked when a
# dataset is built. The articles in the templates ("a", "the") fit every object here.
NAMES = (
    "Mary John Tom James Dan Sid Martin Amy Paul Kate Lisa Mark Sarah Michael Anna "
    "Laura Steven David Robert Emma Jessica Chris Alex Ben Jack Rachel Andrew Kevin "
    "Brian Matt Sam Peter Daniel Jennifer Nicole Patrick Ryan Eric Jason Adam Tim Joe "
    "Rose Helen Alice Bob Jane Susan George Frank Grace Victoria Arthur Henry Thomas Max"
).split()
PLACES = "store garden restaurant school hospital office house station park beach market library".split()
OBJECTS = "ring bone basketball computer necklace drink snack book ball cake bag letter gift".split()

# {n1} and {n2} are the two names in the first clause, {s2} the subject of the second.
# IOI, ABBA: n1 = IO, n2 = S, s2 = S.  BABA: n1 = S, n2 = IO, s2 = S.
# ABC (for means): three different names. The prompt stops right before the answer.
TEMPLATES = (
    "Then, {n1} and {n2} went to the {place}. {s2} gave a {obj} to",
    "Then, {n1} and {n2} had a lot of fun at the {place}. {s2} gave a {obj} to",
    "Then, {n1} and {n2} were working at the {place}. {s2} decided to give a {obj} to",
    "Then, {n1} and {n2} were thinking about going to the {place}. {s2} wanted to give a {obj} to",
    "Then, {n1} and {n2} had a long argument, and afterwards {s2} said to",
    "After {n1} and {n2} went to the {place}, {s2} gave a {obj} to",
    "When {n1} and {n2} got a {obj} at the {place}, {s2} decided to give it to",
    "When {n1} and {n2} got a {obj} at the {place}, {s2} decided to give the {obj} to",
    "While {n1} and {n2} were working at the {place}, {s2} gave a {obj} to",
    "While {n1} and {n2} were commuting to the {place}, {s2} gave a {obj} to",
    "After the lunch, {n1} and {n2} went to the {place}. {s2} gave a {obj} to",
    "Afterwards, {n1} and {n2} went to the {place}. {s2} gave a {obj} to",
    "Then, {n1} and {n2} had a long argument. Afterwards {s2} said to",
    "The {place} {n1} and {n2} went to had a {obj}. {s2} gave it to",
    "Friends {n1} and {n2} found a {obj} at the {place}. {s2} gave it to",
)

# Layer and head counts of models we know, so answers can be checked before any model
# is loaded. Anything else must give config.n_layers and config.n_heads.
KNOWN_SHAPES = {"gpt2": (12, 12), "openai-community/gpt2": (12, 12)}

METRICS = ("faithfulness", "distance")
HEAD_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
ANSWER_FORMAT = 'Heads as "L.H" (layer.head), separated by commas, spaces, or newlines.'


class Prompt(NamedTuple):
    text: str
    template: int
    order: str       # "ABBA", "BABA", or "ABC"
    io: str          # indirect object (the answer). For ABC, the first-clause name n1.
    s: str           # subject. For ABC, n2.


def _fill(template: str, n1: str, n2: str, s2: str, place: str, obj: str) -> str:
    return template.format(n1=n1, n2=n2, s2=s2, place=place, obj=obj)


def generate_ioi(seed: int, n_per_template: int, exclude: frozenset = frozenset()) -> list[Prompt]:
    """IOI prompts: for each template, `n_per_template` in ABBA order and as many in
    BABA order. Deterministic in `seed`. Texts in `exclude` are skipped and redrawn."""
    rng = random.Random(seed)
    out: list[Prompt] = []
    seen: set[str] = set()
    for t, template in enumerate(TEMPLATES):
        for order in ("ABBA", "BABA"):
            made = 0
            while made < n_per_template:
                io, s = rng.sample(NAMES, 2)
                place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
                n1, n2 = (io, s) if order == "ABBA" else (s, io)
                text = _fill(template, n1, n2, s, place, obj)
                if text in exclude or text in seen:
                    continue
                seen.add(text)
                out.append(Prompt(text, t, order, io, s))
                made += 1
    return out


def generate_abc(seed: int, n_per_template: int) -> list[Prompt]:
    """The mean-ablation reference: same templates, three different names."""
    rng = random.Random(seed)
    out: list[Prompt] = []
    for t, template in enumerate(TEMPLATES):
        for _ in range(n_per_template):
            a, b, c = rng.sample(NAMES, 3)
            place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
            out.append(Prompt(_fill(template, a, b, c, place, obj), t, "ABC", a, b))
    return out


def public_prompts(n_per_template: int = 6) -> list[Prompt]:
    return generate_ioi(PUBLIC_SEED, n_per_template)


def parse_heads(answer: str, n_layers: int = 12, n_heads: int = 12) -> list[tuple[int, int]]:
    """Parse "L.H" heads, sorted by (layer, head). Raises ValidationError on bad input."""
    parts = [p for p in re.split(r"[\s,]+", (answer or "").strip()) if p]
    heads: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for p in parts:
        m = HEAD_RE.match(p)
        if not m:
            raise ValidationError(f'"{p[:20]}" is not a head. {ANSWER_FORMAT} '
                                  f'For example: 9.9, 9.6, 10.0')
        layer, head = int(m.group(1)), int(m.group(2))
        if not 0 <= layer < n_layers:
            raise ValidationError(f"Head {p}: the layer must be 0-{n_layers - 1}.")
        if not 0 <= head < n_heads:
            raise ValidationError(f"Head {p}: the head must be 0-{n_heads - 1}.")
        if (layer, head) in seen:
            raise ValidationError(f"Head {layer}.{head} is listed twice.")
        seen.add((layer, head))
        heads.append((layer, head))
    return sorted(heads)


def canonical(heads) -> str:
    return ",".join(f"{layer}.{head}" for layer, head in sorted(heads))


# ── The scorer: built once per challenge, kept in memory ─────────────────────

class _Dataset(NamedTuple):
    input_ids: "object"      # torch (n, T) long, right padded
    last_pos: "object"       # torch (n,) long
    template: "object"       # torch (n,) long, index into the means
    io_tok: "object"         # torch (n,) long
    s_tok: "object"          # torch (n,) long


class CircuitScorer:
    """Holds the model, the prompts as tensors, the means, and the two baselines."""

    def __init__(self, lm, prompts: list[Prompt], abc: list[Prompt], cache_path: Path | None):
        import torch

        self.lm = lm
        self.prompts = prompts
        self.n_layers, self.n_heads = lm.n_layers, lm.n_heads
        lengths = template_lengths(lm, prompts + abc)
        self.data = self._tensors(prompts, lengths)
        cached = _load_cache(cache_path)
        if cached is not None:
            self.means = torch.from_numpy(cached["means"])
            self.ld_full = float(cached["ld_full"])
            self.ld_ablated = float(cached["ld_ablated"])
        else:
            self.means = self._means(abc, lengths)
            self.ld_full = self.mean_ld(np.zeros((self.n_layers, self.n_heads), dtype=bool))
            self.ld_ablated = self.mean_ld(np.ones((self.n_layers, self.n_heads), dtype=bool))
        if not self.ld_full - self.ld_ablated > 1e-6:
            raise ScoringUnavailable("This model does not do the task on these prompts "
                                     "(LD_full is not above LD_ablated).")
        if cached is None:
            _save_cache(cache_path, means=self.means.numpy(), ld_full=self.ld_full,
                        ld_ablated=self.ld_ablated)

    def _tensors(self, prompts: list[Prompt], lengths: dict[int, int]) -> _Dataset:
        import torch

        width = max(lengths.values())
        pad = self.lm.tokenizer.pad_token_id
        ids = torch.full((len(prompts), width), pad, dtype=torch.long)
        for i, p in enumerate(prompts):
            row = self.lm.token_ids(p.text)
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        return _Dataset(
            input_ids=ids,
            last_pos=torch.tensor([lengths[p.template] - 1 for p in prompts], dtype=torch.long),
            template=torch.tensor([p.template for p in prompts], dtype=torch.long),
            io_tok=torch.tensor([single_token(self.lm, p.io) for p in prompts], dtype=torch.long),
            s_tok=torch.tensor([single_token(self.lm, p.s) for p in prompts], dtype=torch.long),
        )

    def _means(self, abc: list[Prompt], lengths: dict[int, int]):
        """(n_layers, n_templates, T, d_model) position-wise means of every head's
        output over the ABC prompts of each template. Positions past a template's end
        stay zero and are never read for a real token."""
        import torch

        from app import local_model

        width = max(lengths.values())
        n_t = len(TEMPLATES)
        means = torch.zeros((self.n_layers, n_t, width, self.lm.hidden_size), dtype=torch.float32)
        for t in range(n_t):
            rows = [self.lm.token_ids(p.text) for p in abc if p.template == t]
            if not rows:
                raise ValueError(f"no ABC prompts for template {t}")
            ids = torch.tensor(rows, dtype=torch.long)
            z = local_model.capture_head_inputs(self.lm, ids)        # (L, b, T_t, D)
            means[:, t, :ids.shape[1]] = z.mean(dim=1)
        return means

    def per_prompt_ld(self, ablate) -> np.ndarray:
        from app import local_model

        d = self.data
        logits = local_model.head_ablation_logits(self.lm, d.input_ids, d.last_pos, ablate,
                                                  self.means, mean_index=d.template)
        rows = np.arange(logits.shape[0])
        lg = logits.numpy()
        return lg[rows, d.io_tok.numpy()] - lg[rows, d.s_tok.numpy()]

    def mean_ld(self, ablate) -> float:
        return float(np.mean(self.per_prompt_ld(ablate)))

    def ablate_mask(self, keep) -> np.ndarray:
        mask = np.ones((self.n_layers, self.n_heads), dtype=bool)
        for layer, head in keep:
            mask[layer, head] = False
        return mask

    def faithfulness(self, keep) -> tuple[float, float]:
        """(faithfulness, LD_circuit) for the circuit that keeps exactly `keep`."""
        ld = self.mean_ld(self.ablate_mask(keep))
        return (ld - self.ld_ablated) / (self.ld_full - self.ld_ablated), ld


def single_token(lm, word: str) -> int:
    ids = lm.token_ids(" " + word)
    if len(ids) != 1:
        raise ValueError(f'" {word}" is {len(ids)} tokens for {lm.model_id}, not one')
    return ids[0]


def template_lengths(lm, prompts: list[Prompt]) -> dict[int, int]:
    """Token length of each template. Raises if two prompts of a template differ, which
    would mean a filler is not a single token and position-wise means would misalign."""
    for w in NAMES:
        single_token(lm, w)
    lengths: dict[int, int] = {}
    for p in prompts:
        n = len(lm.token_ids(p.text))
        if lengths.setdefault(p.template, n) != n:
            raise ValueError(f"template {p.template} gives prompts of {lengths[p.template]} "
                             f"and {n} tokens")
    return lengths


def _load_cache(path: Path | None):
    if path is None or not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _save_cache(path: Path | None, **arrays) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
    tmp.replace(path)


# challenge id -> (config fingerprint, scorer). Built on first use, like steering.
_SCORERS: dict[int, tuple[str, CircuitScorer]] = {}
_BUILD_LOCK = threading.Lock()


class CircuitsChallenge(Challenge):
    kind = "circuits"
    mode = "score"
    compute = "local"

    def validate_config(self) -> None:
        cfg = self.config
        model_id = cfg.setdefault("model_id", "gpt2")
        if model_id not in KNOWN_SHAPES and not ("n_layers" in cfg and "n_heads" in cfg):
            raise ValueError(f"circuits: unknown model {model_id!r}, give config.n_layers "
                             f"and config.n_heads")
        split = cfg.setdefault("split", "public")
        if split not in ("public", "hidden"):
            raise ValueError('circuits: config.split must be "public" or "hidden"')
        if split == "hidden" and not cfg.get("seed_file"):
            raise ValueError("circuits: a hidden split needs config.seed_file (data/private/...)")
        n_layers, n_heads = self.shape
        if not 1 <= self.max_heads <= n_layers * n_heads:
            raise ValueError(f"circuits: config.max_heads must be 1-{n_layers * n_heads}")
        if cfg.get("metric", "faithfulness") not in METRICS:
            raise ValueError(f"circuits: config.metric must be one of {', '.join(METRICS)}")
        for k in ("n_per_template", "n_mean_per_template"):
            if k in cfg and int(cfg[k]) < 1:
                raise ValueError(f"circuits: config.{k} must be at least 1")

    @property
    def shape(self) -> tuple[int, int]:
        cfg = self.config
        if "n_layers" in cfg and "n_heads" in cfg:
            return int(cfg["n_layers"]), int(cfg["n_heads"])
        return KNOWN_SHAPES[cfg["model_id"]]

    @property
    def metric(self) -> str:
        return self.config.get("metric", "faithfulness")

    @property
    def higher_is_better(self) -> bool:
        return self.metric == "faithfulness"

    @property
    def max_heads(self) -> int:
        return int(self.config.get("max_heads", 30))

    def parse(self, answer: str) -> list[tuple[int, int]]:
        heads = parse_heads(answer, *self.shape)
        if not heads:
            raise ValidationError(f"Submit at least one head. {ANSWER_FORMAT}")
        if len(heads) > self.max_heads:
            raise ValidationError(f"That is {len(heads)} heads. The budget is {self.max_heads}.")
        return heads

    def normalize(self, answer: str) -> str:
        return canonical(self.parse(answer))

    def answer_key(self, answer: str) -> str:
        return self.normalize(answer)

    def prompts(self) -> tuple[list[Prompt], list[Prompt], int]:
        """(IOI prompts, ABC prompts, seed) for this challenge's split."""
        cfg = self.config
        n = int(cfg.get("n_per_template", 6))
        n_mean = int(cfg.get("n_mean_per_template", 32))
        if cfg["split"] == "public":
            seed = PUBLIC_SEED
            exclude: frozenset = frozenset()
        else:
            seed = _read_seed(cfg["seed_file"])
            exclude = frozenset(p.text for p in public_prompts(n))
        # The ABC stream is offset so it never shares draws with the IOI stream.
        return generate_ioi(seed, n, exclude), generate_abc(seed + 1, n_mean), seed

    def _fingerprint(self) -> str:
        return json.dumps(self.config, sort_keys=True)

    def prepare(self, settings, on_remote_call) -> None:
        """Build the scorer on first use. Never calls `on_remote_call`: nothing here
        leaves this server."""
        fp = self._fingerprint()
        hit = _SCORERS.get(self.id)
        if hit is not None and hit[0] == fp:
            return
        with _BUILD_LOCK:
            hit = _SCORERS.get(self.id)
            if hit is not None and hit[0] == fp:
                return
            from app import local_model

            prompts, abc, seed = self.prompts()
            lm = local_model.load(self.config["model_id"], settings.local_device,
                                  settings.local_threads)
            if (lm.n_layers, lm.n_heads) != self.shape:
                raise ScoringUnavailable("The circuits model does not match its config.")
            # The key covers everything the cached numbers depend on. It is a hash, so
            # the file name does not reveal a hidden seed.
            blob = "\n".join([f"{CACHE_VERSION}|{self.config['model_id']}|{seed}",
                              *(p.text for p in prompts), "|", *(p.text for p in abc)])
            key = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
            path = Path(settings.cache_dir) / f"circuits_{self.slug}_{key}.npz"
            _SCORERS[self.id] = (fp, CircuitScorer(lm, prompts, abc, path))

    def evaluate(self, answer: str) -> Outcome:
        heads = self.parse(answer)
        _fp, scorer = _SCORERS[self.id]
        faith, ld = scorer.faithfulness(heads)
        score = faith if self.metric == "faithfulness" else abs(1.0 - faith)
        return Outcome(correct=None, score=float(score), detail={
            "heads": len(heads), "faithfulness": round(faith, 6), "ld_circuit": round(ld, 4),
            "ld_full": round(scorer.ld_full, 4), "ld_ablated": round(scorer.ld_ablated, 4)})

    def public_config(self) -> dict:
        cfg = self.config
        out = {"model_id": cfg["model_id"], "split": cfg["split"], "max_heads": self.max_heads,
               "metric": self.metric, "higher_is_better": self.higher_is_better,
               "answer_format": ANSWER_FORMAT,
               "n_layers": self.shape[0], "n_heads": self.shape[1],
               "n_prompts": int(cfg.get("n_per_template", 6)) * 2 * len(TEMPLATES)}
        if cfg.get("prompts_url"):
            out["prompts_url"] = cfg["prompts_url"]
        return out


def _read_seed(path: str) -> int:
    p = Path(path)
    if not p.exists():
        raise ScoringUnavailable("This challenge's hidden prompts are not installed on the "
                                 "server yet.")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return int(obj["seed"] if isinstance(obj, dict) else obj)


def reset_scorers() -> None:
    _SCORERS.clear()
