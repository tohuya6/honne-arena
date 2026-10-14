"""Small Hugging Face causal LMs (GPT-2 small) run on this server, for the learn tiers.

Nothing here touches NDIF. A model is loaded once per (model_id, device), kept in eval
mode with gradients off, and every forward runs under torch.inference_mode.

Hooks are module state shared by every caller of a model, so each forward that uses them
holds that model's lock. Two local jobs in flight at once therefore take turns on one
model instead of seeing each other's hooks.

What "layer L" means: the residual stream at the output of decoder block L, which is
what `ResidualReader` reads on NDIF (`model.model.layers[L].output`). For GPT-2 that is
the output of `transformer.h[L]`, before the final layer norm. It equals
`hidden_states[L + 1]` from `output_hidden_states=True` for every block but the last
(the last entry of `hidden_states` has `ln_f` applied).

Heavy deps (torch, transformers) are imported lazily, like `app.ndif_client`.
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

_LOAD_LOCK = threading.Lock()
_MODELS: dict[tuple[str, str], "LocalModel"] = {}


class LocalModel:
    """A loaded model, its tokenizer, and the lock that serializes its forwards."""

    def __init__(self, model, tokenizer, model_id: str, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.lock = threading.Lock()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    @property
    def config(self):
        return self.model.config

    @property
    def hidden_size(self) -> int:
        return int(self.config.n_embd)

    @property
    def n_layers(self) -> int:
        return int(self.config.n_layer)

    @property
    def n_heads(self) -> int:
        return int(self.config.n_head)

    @property
    def d_head(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def blocks(self):
        """The decoder blocks. GPT-2 layout only (`transformer.h`)."""
        return self.model.transformer.h

    def token_ids(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])


def _prepare(model, tokenizer, model_id: str, device: str) -> LocalModel:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError(f"{model_id} is not a GPT-2 style model (no transformer.h)")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return LocalModel(model, tokenizer, model_id, device)


def load(model_id: str, device: str = "cpu") -> LocalModel:
    """The cached model for `model_id` on `device`, loading it on first use.

    Uses the eager attention path: it is deterministic on CPU, and its additive mask
    never turns a fully masked padding row into NaN the way some fused kernels can."""
    key = (model_id, device)
    lm = _MODELS.get(key)
    if lm is not None:
        return lm
    with _LOAD_LOCK:
        lm = _MODELS.get(key)
        if lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager")
            lm = _MODELS[key] = _prepare(model, tokenizer, model_id, device)
    return lm


def register(model_id: str, model, tokenizer, device: str = "cpu") -> LocalModel:
    """Put an already built model in the cache under `model_id`. Tests use this with a
    tiny random GPT-2 so they never download anything."""
    lm = _prepare(model, tokenizer, model_id, device)
    with _LOAD_LOCK:
        _MODELS[(model_id, device)] = lm
    return lm


def clear() -> None:
    with _LOAD_LOCK:
        _MODELS.clear()


# ── Last-token residuals (the steering read) ─────────────────────────────────

class _Stop(Exception):
    """Raised from a hook to skip the blocks after the one we read."""


def _left_pad(lm: LocalModel, rows: Sequence[Sequence[int]]):
    import torch

    width = max(len(r) for r in rows)
    pad = lm.tokenizer.pad_token_id
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, width - len(r):] = torch.tensor(list(r), dtype=torch.long)
        mask[i, width - len(r):] = 1
    # GPT-2 has absolute position embeddings and its forward does not derive positions
    # from the mask, so left padding needs explicit positions. Each row's first real
    # token gets position 0, exactly as if it ran alone.
    pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
    return ids, mask, pos


def batch_last_resids(lm: LocalModel, texts: Sequence[str], layer: int, *,
                      prepend_bos: bool = False) -> np.ndarray:
    """Residual at the output of block `layer`, at each text's last token, for a batch
    of texts in one forward. Returns float32 (n, hidden).

    Same contract as `ResidualReader.batch_last_resids`, so `app.scoring` works on it
    unchanged. Rows are left padded so the last real token sits at index -1, padding is
    masked out, and positions restart at 0 on each row's first real token, so a row
    matches the value for that text alone.

    `prepend_bos` puts the tokenizer's BOS token (GPT-2's <|endoftext|>) in front of
    every text. The tokenizer default for GPT-2 adds none."""
    import torch

    texts = list(texts)
    if not texts:
        raise ValueError("batch_last_resids needs at least one text")
    if not 0 <= int(layer) < lm.n_layers:
        raise ValueError(f"layer {layer} is out of range for {lm.n_layers} layers")
    rows = []
    for t in texts:
        ids = lm.token_ids(t)
        if prepend_bos:
            ids = [lm.tokenizer.bos_token_id] + ids
        if not ids:
            raise ValueError("cannot read the residual of an empty text")
        rows.append(ids)
    ids, mask, pos = _left_pad(lm, rows)

    saved = {}

    def grab(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        saved["v"] = hidden[:, -1, :].to(torch.float32).cpu().numpy()
        raise _Stop

    with lm.lock, torch.inference_mode():
        handle = lm.blocks[int(layer)].register_forward_hook(grab)
        try:
            lm.model.transformer(input_ids=ids.to(lm.device), attention_mask=mask.to(lm.device),
                                 position_ids=pos.to(lm.device), use_cache=False)
        except _Stop:
            pass
        finally:
            handle.remove()
    return np.asarray(saved["v"], dtype=np.float32)


# ── Attention head outputs and head ablation (the circuits forward) ──────────
#
# GPT-2's attention concatenates its heads into one (batch, seq, n_heads * d_head)
# tensor and multiplies it by `attn.c_proj`. Head h owns the slice
# [h * d_head, (h + 1) * d_head) of that input, so replacing the slice replaces exactly
# that head's contribution to the residual stream (bias included, it is shared).

def capture_head_inputs(lm: LocalModel, input_ids) -> "torch.Tensor":
    """The `attn.c_proj` input of every layer for one batch: float32
    (n_layers, batch, seq, n_heads * d_head). Used to build mean activations.

    No attention mask, so pass rows of equal length or right padding (a causal model
    never lets a real token see padding that comes after it)."""
    import torch

    out = [None] * lm.n_layers
    handles = []

    def make(layer):
        def pre(_module, args):
            out[layer] = args[0].detach().to(torch.float32).clone()
        return pre

    with lm.lock, torch.inference_mode():
        try:
            for layer, block in enumerate(lm.blocks):
                handles.append(block.attn.c_proj.register_forward_pre_hook(make(layer)))
            lm.model.transformer(input_ids=input_ids.to(lm.device), use_cache=False)
        finally:
            for h in handles:
                h.remove()
    return torch.stack(out).cpu()


def head_ablation_logits(lm: LocalModel, input_ids, last_pos, ablate, means,
                         mean_index=None):
    """Logits at `last_pos` of each row, with the heads in `ablate` replaced by means.

    input_ids   (batch, seq) long. Right padded if rows differ in length.
    last_pos    (batch,) long, the position whose next-token logits we want.
    ablate      (n_layers, n_heads) bool. True means replace that head's output.
    means       (n_layers, M, T, n_heads * d_head) float, T >= seq. The replacement
                for row i is means[layer, mean_index[i], :seq], at every position.
    mean_index  (batch,) long into M. Defaults to 0 for every row.

    Returns float32 (batch, vocab). Every hook is removed before returning, also when
    the forward raises."""
    import torch

    ablate = torch.as_tensor(np.asarray(ablate, dtype=bool))
    if tuple(ablate.shape) != (lm.n_layers, lm.n_heads):
        raise ValueError(f"ablate must be ({lm.n_layers}, {lm.n_heads}), got {tuple(ablate.shape)}")
    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    batch, seq = input_ids.shape
    last_pos = torch.as_tensor(last_pos, dtype=torch.long)
    if mean_index is None:
        mean_index = torch.zeros(batch, dtype=torch.long)
    mean_index = torch.as_tensor(mean_index, dtype=torch.long)
    n_heads, d_head = lm.n_heads, lm.d_head

    def make(layer):
        drop = ablate[layer].to(lm.device).view(1, 1, n_heads, 1)

        def pre(_module, args):
            z = args[0]
            m = means[layer, mean_index, :seq].to(device=z.device, dtype=z.dtype)
            z4 = z.view(batch, seq, n_heads, d_head)
            mixed = torch.where(drop, m.view(batch, seq, n_heads, d_head), z4)
            return (mixed.reshape(batch, seq, n_heads * d_head),)
        return pre

    handles = []
    with lm.lock, torch.inference_mode():
        try:
            for layer, block in enumerate(lm.blocks):
                if bool(ablate[layer].any()):
                    handles.append(block.attn.c_proj.register_forward_pre_hook(make(layer)))
            hidden = lm.model.transformer(input_ids=input_ids.to(lm.device),
                                          use_cache=False).last_hidden_state
            rows = torch.arange(batch, device=hidden.device)
            final = hidden[rows, last_pos.to(hidden.device)]
            logits = lm.model.lm_head(final)
        finally:
            for h in handles:
                h.remove()
    return logits.to(torch.float32).cpu()
