"""app.local_model on a tiny random GPT-2, built in memory (no download).

The point: a batched, left-padded read gives the same row as the text alone, "layer L"
is the output of block L, head ablation replaces exactly the chosen heads, and every
hook is gone afterwards."""

from __future__ import annotations

import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from app import local_model  # noqa: E402
from app.challenges import circuits  # noqa: E402

TINY_ID = "tiny-gpt2-test"
N_LAYERS, N_HEADS, D_MODEL = 2, 4, 32

EXTRA_WORDS = """the a an of to and is was it this that i you he she we they in on at for
with my our your hello world good bad great terrible love hate happy sad today day
here list items box train leaves station make sauce first report covers period from""".split()


def tiny_tokenizer():
    """A word-level tokenizer over every word the tests and the IOI templates use, so
    each name, place, and object is one token, as in GPT-2."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = set(EXTRA_WORDS) | set(circuits.NAMES) | set(circuits.PLACES) | set(circuits.OBJECTS)
    for t in circuits.TEMPLATES:
        words |= set(re.findall(r"\w+|[^\w\s]+", re.sub(r"\{\w+\}", " ", t)))
    vocab = {"<|endoftext|>": 0, "[UNK]": 1}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|endoftext|>",
                                   eos_token="<|endoftext|>", unk_token="[UNK]")


def tiny_lm(model_id: str = TINY_ID, seed: int = 0) -> local_model.LocalModel:
    """Register a tiny random GPT-2 under `model_id` and return it."""
    tok = tiny_tokenizer()
    cfg = transformers.GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=D_MODEL,
                                  n_layer=N_LAYERS, n_head=N_HEADS)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = transformers.GPT2LMHeadModel(cfg)
    return local_model.register(model_id, model, tok)


def require_cached_gpt2() -> None:
    """Skip unless GPT-2 small is already in the local Hugging Face cache, so a test run
    never starts a 500 MB download (set HF_HOME to where it lives)."""
    from huggingface_hub import try_to_load_from_cache

    for name in ("config.json", "model.safetensors"):
        if not isinstance(try_to_load_from_cache("gpt2", name), str):
            pytest.skip("gpt2 is not in the local Hugging Face cache (set HF_HOME)")


@pytest.fixture
def lm():
    m = tiny_lm()
    yield m
    local_model.clear()


def _no_hooks(lm) -> bool:
    for block in lm.blocks:
        if block._forward_hooks or block._forward_pre_hooks:
            return False
        if block.attn.c_proj._forward_pre_hooks or block.attn.c_proj._forward_hooks:
            return False
    return True


TEXTS = ["hello world", "the train leaves the station at", "good"]


def test_register_then_load_returns_the_same_model(lm):
    assert local_model.load(TINY_ID, "cpu") is lm


def test_batched_left_padded_rows_match_each_text_alone(lm):
    for layer in range(N_LAYERS):
        batch = local_model.batch_last_resids(lm, TEXTS, layer)
        assert batch.shape == (len(TEXTS), D_MODEL) and batch.dtype == np.float32
        for i, t in enumerate(TEXTS):
            alone = local_model.batch_last_resids(lm, [t], layer)[0]
            np.testing.assert_allclose(batch[i], alone, rtol=1e-5, atol=1e-5)


def test_layer_is_the_output_of_that_block(lm):
    ids = torch.tensor([lm.token_ids("the train leaves the station at")])
    with torch.inference_mode():
        hs = lm.model(input_ids=ids, output_hidden_states=True).hidden_states
    got0 = local_model.batch_last_resids(lm, ["the train leaves the station at"], 0)[0]
    np.testing.assert_allclose(got0, hs[1][0, -1].numpy(), rtol=1e-5, atol=1e-5)

    # The last block's output is read before ln_f, so it is NOT hidden_states[-1].
    captured = {}

    def grab(_m, _a, o):
        captured["v"] = (o[0] if isinstance(o, tuple) else o)[0, -1].clone()

    h = lm.blocks[-1].register_forward_hook(grab)
    with torch.inference_mode():
        lm.model(input_ids=ids)
    h.remove()
    last = local_model.batch_last_resids(lm, ["the train leaves the station at"], N_LAYERS - 1)[0]
    np.testing.assert_allclose(last, captured["v"].numpy(), rtol=1e-5, atol=1e-5)
    assert not np.allclose(last, hs[-1][0, -1].numpy(), atol=1e-3)


def test_prepend_bos_reads_the_text_after_a_bos_token(lm):
    with_bos = local_model.batch_last_resids(lm, ["hello world"], 1, prepend_bos=True)[0]
    explicit = local_model.batch_last_resids(lm, ["<|endoftext|> hello world"], 1)[0]
    plain = local_model.batch_last_resids(lm, ["hello world"], 1)[0]
    np.testing.assert_allclose(with_bos, explicit, rtol=1e-5, atol=1e-5)
    assert not np.allclose(with_bos, plain)


def test_read_removes_its_hook_and_rejects_bad_layers(lm):
    local_model.batch_last_resids(lm, TEXTS, 0)
    assert _no_hooks(lm)
    with pytest.raises(ValueError):
        local_model.batch_last_resids(lm, TEXTS, N_LAYERS)
    with pytest.raises(ValueError):
        local_model.batch_last_resids(lm, [], 0)


def _batch(lm, texts):
    rows = [lm.token_ids(t) for t in texts]
    width = max(len(r) for r in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
    last = torch.tensor([len(r) - 1 for r in rows])
    return ids, last


def _random_means(lm, width, m=1, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((N_LAYERS, m, width, D_MODEL), generator=g)


def test_no_ablation_equals_the_plain_forward(lm):
    ids, last = _batch(lm, TEXTS)
    means = _random_means(lm, ids.shape[1])
    got = local_model.head_ablation_logits(lm, ids, last, np.zeros((N_LAYERS, N_HEADS), bool), means)
    for i, t in enumerate(TEXTS):
        alone = torch.tensor([lm.token_ids(t)])
        with torch.inference_mode():
            ref = lm.model(input_ids=alone).logits[0, -1]
        np.testing.assert_allclose(got[i].numpy(), ref.numpy(), rtol=1e-4, atol=1e-4)
    assert _no_hooks(lm)


def test_ablating_every_head_gives_the_mean_only_attention_output(lm):
    ids, last = _batch(lm, TEXTS)
    means = _random_means(lm, ids.shape[1] + 3)          # longer than the batch is fine
    seen = {}

    def make(i):
        def grab(_m, a, o):
            seen[i] = (a[0].clone(), o.clone())
        return grab

    handles = [b.attn.c_proj.register_forward_hook(make(i)) for i, b in enumerate(lm.blocks)]
    try:
        local_model.head_ablation_logits(lm, ids, last, np.ones((N_LAYERS, N_HEADS), bool), means)
    finally:
        for h in handles:
            h.remove()
    for layer, block in enumerate(lm.blocks):
        z, out = seen[layer]
        expect_z = means[layer, 0, :ids.shape[1]].expand(len(TEXTS), -1, -1).contiguous()
        np.testing.assert_allclose(z.numpy(), expect_z.numpy(), rtol=1e-6, atol=1e-6)
        with torch.inference_mode():
            expect_out = block.attn.c_proj(expect_z)
        np.testing.assert_allclose(out.numpy(), expect_out.numpy(), rtol=1e-5, atol=1e-5)
    assert _no_hooks(lm)


def test_one_ablated_head_only_touches_its_own_slice(lm):
    ids, last = _batch(lm, TEXTS)
    means = _random_means(lm, ids.shape[1])
    ablate = np.zeros((N_LAYERS, N_HEADS), bool)
    ablate[1, 2] = True
    seen = {}

    def grab(_m, a, _o):
        seen["z"] = a[0].clone()

    h = lm.blocks[1].attn.c_proj.register_forward_hook(grab)
    try:
        got = local_model.head_ablation_logits(lm, ids, last, ablate, means)
    finally:
        h.remove()
    clean = local_model.capture_head_inputs(lm, ids)[1]
    d = D_MODEL // N_HEADS
    z = seen["z"]
    np.testing.assert_allclose(z[..., 2 * d:3 * d].numpy(),
                               means[1, 0, :ids.shape[1]].expand(len(TEXTS), -1, -1)[..., 2 * d:3 * d].numpy())
    keep = torch.cat([z[..., :2 * d], z[..., 3 * d:]], dim=-1)
    ref = torch.cat([clean[..., :2 * d], clean[..., 3 * d:]], dim=-1)
    np.testing.assert_allclose(keep.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)
    plain = local_model.head_ablation_logits(lm, ids, last, np.zeros_like(ablate), means)
    assert not np.allclose(got.numpy(), plain.numpy())


def test_mean_index_picks_a_row_of_means_per_prompt(lm):
    ids, last = _batch(lm, TEXTS)
    means = _random_means(lm, ids.shape[1], m=3)
    every = np.ones((N_LAYERS, N_HEADS), bool)
    mixed = local_model.head_ablation_logits(lm, ids, last, every, means,
                                             mean_index=torch.tensor([2, 0, 1]))
    for i, j in enumerate([2, 0, 1]):
        single = local_model.head_ablation_logits(lm, ids[i:i + 1], last[i:i + 1], every,
                                                  means[:, j:j + 1])
        np.testing.assert_allclose(mixed[i].numpy(), single[0].numpy(), rtol=1e-4, atol=1e-4)


def test_hooks_are_removed_when_the_forward_fails(lm):
    ids, last = _batch(lm, TEXTS)
    too_short = _random_means(lm, 1)                      # cannot cover the sequence
    with pytest.raises(RuntimeError):
        local_model.head_ablation_logits(lm, ids, last, np.ones((N_LAYERS, N_HEADS), bool),
                                         too_short)
    assert _no_hooks(lm)
    with pytest.raises(ValueError):
        local_model.head_ablation_logits(lm, ids, last, np.ones((3, 3), bool), too_short)


def test_capture_head_inputs_shape_and_cleanup(lm):
    ids, _last = _batch(lm, TEXTS)
    z = local_model.capture_head_inputs(lm, ids)
    assert tuple(z.shape) == (N_LAYERS, len(TEXTS), ids.shape[1], D_MODEL)
    assert _no_hooks(lm)


def test_limit_threads_caps_torch():
    import torch

    from app import local_model

    before = torch.get_num_threads()
    try:
        local_model.limit_threads(2)
        assert torch.get_num_threads() == 2
        local_model.limit_threads(0)          # 0 leaves the current setting alone
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)
