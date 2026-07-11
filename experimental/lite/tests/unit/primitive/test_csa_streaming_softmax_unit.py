# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Streaming-softmax CP path for Compressed Sparse Attention (DSA).

The CP/training path replaces a monolithic ``[b, heads, q, Sum(key_len)]`` score
matrix (which OOMs past ~32K context) with a chunked online-softmax that keeps
only O(seq_len) running state ``(max, denom, numer)`` and recomputes scores +
the ``part_max``/``new_max``/``exp_old`` intermediates in backward.

These tests pin down that the memory optimization is *numerically* free:

* each ``torch.autograd.Function`` matches an fp64 autograd reference of the same
  math, in both single-chunk and forced multi-chunk regimes;
* chaining several dense updates on top of a finite ``sink`` seed reproduces one
  monolithic softmax over the concatenated scores/values (the actual CP identity);
* fully-masked score blocks and tied maxima stay finite (no ``-inf - (-inf)``);
* the forward helper / backward argument-and-return arities stay balanced
  (guards the two regressions found during review).
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from megatron.lite.primitive.modules.attention import csa

pytestmark = pytest.mark.mlite

# --- small fixed problem sizes; large enough to force many chunks when asked ---
B, H, Q, D, KEY = 2, 3, 24, 16, 20
IDX_H, IDX_D = 2, 8
RATIO, TOPK = 4, 8
SCALE = D**-0.5
NO_WINDOW = 1_000_000  # effectively disables sliding-window masking

# fp32 internal precision (the Functions call .float()) vs an fp64 reference:
# grads reach magnitude ~20, so ~1e-6 absolute == ~1e-7 relative. Keep headroom.
RTOL, ATOL = 2e-3, 1e-4


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _positions():
    qpos = torch.arange(Q).unsqueeze(0).expand(B, Q).clone()
    kpos = torch.arange(KEY).unsqueeze(0).expand(B, KEY).clone()
    comp_pos = (torch.arange(KEY) * RATIO).unsqueeze(0).expand(B, KEY).clone()
    return qpos, kpos, comp_pos


def _state_master():
    # High-precision master values for the running softmax state.
    return {
        "ms": torch.randn(B, H, Q, 1, dtype=torch.float64),
        "dn": torch.rand(B, H, Q, 1, dtype=torch.float64) + 0.5,
        "nu": torch.randn(B, H, Q, D, dtype=torch.float64),
    }


def _make_pair(master):
    """The Functions run fp32 internally; the reference runs fp64. Build a leaf
    set for each from the same master values so their grads are comparable."""
    custom = {k: v.detach().float().clone().requires_grad_(True) for k, v in master.items()}
    ref = {k: v.detach().double().clone().requires_grad_(True) for k, v in master.items()}
    return custom, ref


def _force_chunk(monkeypatch, size):
    monkeypatch.setattr(csa, "_streaming_q_chunk", lambda q_len, denom_elems: size)


def _online_update_ref(scores, values, max_scores, denom, numer):
    """Single-shot online-softmax update; ``scores`` already -inf-masked."""
    part_max = scores.max(dim=-1, keepdim=True).values
    new_max = torch.maximum(max_scores, part_max)
    exp_old = torch.exp(max_scores - new_max)
    exp_part = torch.exp(scores - new_max)
    new_denom = denom * exp_old + exp_part.sum(dim=-1, keepdim=True)
    new_numer = numer * exp_old + torch.matmul(exp_part, values)
    return new_max, new_denom, new_numer


def _backprop(outs, seed):
    (outs[0] * seed[0] + outs[1] * seed[1] + (outs[2] * seed[2]).sum(-1, keepdim=True)).sum().backward()


def _grad(tensor: torch.Tensor) -> torch.Tensor:
    """Return .grad, asserting it was populated (also narrows Tensor|None for type checkers)."""
    assert tensor.grad is not None
    return tensor.grad


def _compare(custom_leaves, ref_leaves, custom_outs, ref_outs, names):
    torch.manual_seed(1234)
    seed = [torch.randn_like(o, dtype=torch.float64) for o in ref_outs]
    _backprop(custom_outs, [s.to(o.dtype) for s, o in zip(seed, custom_outs)])
    _backprop(ref_outs, seed)
    for o_c, o_r in zip(custom_outs, ref_outs):
        _assert_close(o_c.double(), o_r)
    for name in names:
        _assert_close(_grad(custom_leaves[name]).double(), _grad(ref_leaves[name]))


# --------------------------------------------------------------------------- #
# per-Function equivalence vs fp64 autograd reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk", [Q, 3], ids=["single_chunk", "multi_chunk"])
def test_dense_matches_autograd(monkeypatch, chunk):
    _force_chunk(monkeypatch, chunk)
    qpos, kpos, _ = _positions()
    c, r = _make_pair(_state_master())
    q_m = torch.randn(B, H, Q, D, dtype=torch.float64)
    src_m = torch.randn(B, H, KEY, D, dtype=torch.float64)
    c["q"] = q_m.float().clone().requires_grad_(True)
    c["src"] = src_m.float().clone().requires_grad_(True)
    r["q"] = q_m.clone().requires_grad_(True)
    r["src"] = src_m.clone().requires_grad_(True)

    custom_outs = csa._StreamingSoftmaxDenseUpdateFn.apply(
        c["ms"], c["dn"], c["nu"], c["q"], c["src"], qpos, kpos, SCALE, NO_WINDOW
    )
    scores = torch.matmul(r["q"], r["src"].transpose(-1, -2)) * SCALE
    mask = csa._source_scores_mask(qpos, kpos, sliding_window=NO_WINDOW).unsqueeze(1)
    ref_outs = _online_update_ref(
        scores.masked_fill(~mask, -float("inf")), r["src"], r["ms"], r["dn"], r["nu"]
    )
    _compare(c, r, custom_outs, ref_outs, ["ms", "dn", "nu", "q", "src"])


@pytest.mark.parametrize("chunk", [Q, 3], ids=["single_chunk", "multi_chunk"])
def test_compressed_matches_autograd(monkeypatch, chunk):
    _force_chunk(monkeypatch, chunk)
    qpos, _, comp_pos = _positions()
    c, r = _make_pair(_state_master())
    q_m = torch.randn(B, H, Q, D, dtype=torch.float64)
    comp_m = torch.randn(B, H, KEY, D, dtype=torch.float64)
    c["q"] = q_m.float().clone().requires_grad_(True)
    c["comp"] = comp_m.float().clone().requires_grad_(True)
    r["q"] = q_m.clone().requires_grad_(True)
    r["comp"] = comp_m.clone().requires_grad_(True)

    custom_outs = csa._StreamingSoftmaxCompressedUpdateFn.apply(
        c["ms"], c["dn"], c["nu"], c["q"], c["comp"], qpos, comp_pos, RATIO, SCALE
    )
    scores = torch.matmul(r["q"], r["comp"].transpose(-1, -2)) * SCALE
    valid = csa._compressed_scores_mask(qpos, comp_pos, ratio=RATIO).unsqueeze(1)
    ref_outs = _online_update_ref(
        scores.masked_fill(~valid, -float("inf")), r["comp"], r["ms"], r["dn"], r["nu"]
    )
    _compare(c, r, custom_outs, ref_outs, ["ms", "dn", "nu", "q", "comp"])


@pytest.mark.parametrize("chunk", [Q, 3], ids=["single_chunk", "multi_chunk"])
def test_indexed_matches_autograd(monkeypatch, chunk):
    _force_chunk(monkeypatch, chunk)
    qpos, _, comp_pos = _positions()
    c, r = _make_pair(_state_master())
    masters = {
        "q": torch.randn(B, H, Q, D, dtype=torch.float64),
        "comp": torch.randn(B, H, KEY, D, dtype=torch.float64),
        "q_idx": torch.randn(B, IDX_H, Q, IDX_D, dtype=torch.float64),
        "k_idx": torch.randn(B, KEY, IDX_D, dtype=torch.float64),
        "w": torch.randn(B, Q, IDX_H, dtype=torch.float64),
    }
    for k, v in masters.items():
        c[k] = v.float().clone().requires_grad_(True)
        r[k] = v.clone().requires_grad_(True)

    custom_outs = csa._StreamingSoftmaxCompressedIndexedUpdateFn.apply(
        c["ms"], c["dn"], c["nu"], c["q"], c["comp"], qpos, comp_pos,
        c["q_idx"], c["k_idx"], c["w"], TOPK, RATIO, SCALE,
    )
    scores = torch.matmul(r["q"], r["comp"].transpose(-1, -2)) * SCALE
    valid = csa._compressed_scores_mask(qpos, comp_pos, ratio=RATIO)
    z = torch.einsum("bhsd,btd->bsht", r["q_idx"], r["k_idx"]).relu()
    index = (z * r["w"].unsqueeze(-1)).sum(dim=2).masked_fill(~valid, -float("inf"))
    topk_idx = index.topk(min(TOPK, KEY), dim=-1).indices
    topk_mask = torch.zeros_like(index, dtype=torch.bool).scatter_(-1, topk_idx, True)
    scores = (scores + index.unsqueeze(1)).masked_fill(~(valid & topk_mask).unsqueeze(1), -float("inf"))
    ref_outs = _online_update_ref(scores, r["comp"], r["ms"], r["dn"], r["nu"])
    _compare(c, r, custom_outs, ref_outs, ["ms", "dn", "nu", "q", "comp", "q_idx", "k_idx", "w"])


# --------------------------------------------------------------------------- #
# the actual CP identity: chained streaming == one monolithic softmax
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk", [Q, 3], ids=["single_chunk", "multi_chunk"])
def test_chained_dense_matches_monolithic(monkeypatch, chunk):
    """init(sink) -> dense(A) -> dense(B) must equal softmax over
    concat([sink | scoresA | scoresB]) with values concat([0 | A | B])."""
    _force_chunk(monkeypatch, chunk)
    qpos, kpos, _ = _positions()
    masters = {
        "q": torch.randn(B, H, Q, D, dtype=torch.float64),
        "srcA": torch.randn(B, H, KEY, D, dtype=torch.float64),
        "srcB": torch.randn(B, H, KEY, D, dtype=torch.float64),
        "sink": torch.randn(B, H, Q, 1, dtype=torch.float64),
    }
    c = {k: v.float().clone().requires_grad_(True) for k, v in masters.items()}
    r = {k: v.clone().requires_grad_(True) for k, v in masters.items()}

    # streaming (fp32): seed running state from the finite sink term, then chain
    den0 = torch.ones(B, H, Q, 1)
    num0 = torch.zeros(B, H, Q, D)
    m, d, n = csa._StreamingSoftmaxDenseUpdateFn.apply(
        c["sink"], den0, num0, c["q"], c["srcA"], qpos, kpos, SCALE, NO_WINDOW
    )
    m, d, n = csa._StreamingSoftmaxDenseUpdateFn.apply(
        m, d, n, c["q"], c["srcB"], qpos, kpos, SCALE, NO_WINDOW
    )
    ctx_stream = n / d  # softmax-weighted value

    # monolithic fp64 reference
    mask = csa._source_scores_mask(qpos, kpos, sliding_window=NO_WINDOW).unsqueeze(1)
    sa = (torch.matmul(r["q"], r["srcA"].transpose(-1, -2)) * SCALE).masked_fill(~mask, -float("inf"))
    sb = (torch.matmul(r["q"], r["srcB"].transpose(-1, -2)) * SCALE).masked_fill(~mask, -float("inf"))
    all_scores = torch.cat([r["sink"], sa, sb], dim=-1)
    zeros_v = torch.zeros(B, H, 1, D, dtype=torch.float64)
    all_values = torch.cat([zeros_v, r["srcA"], r["srcB"]], dim=-2)
    weights = torch.softmax(all_scores, dim=-1)
    ctx_ref = torch.matmul(weights, all_values)

    _assert_close(ctx_stream.double(), ctx_ref)
    torch.manual_seed(99)
    g = torch.randn_like(ctx_ref)
    (ctx_stream * g.to(ctx_stream.dtype)).sum().backward()
    (ctx_ref * g).sum().backward()
    for name in ["q", "srcA", "srcB", "sink"]:
        _assert_close(_grad(c[name]).double(), _grad(r[name]))


# --------------------------------------------------------------------------- #
# edge cases: fully-masked block and tied maxima must stay finite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tie", [False, True], ids=["masked_block", "ties"])
def test_edge_cases_finite(monkeypatch, tie):
    _force_chunk(monkeypatch, 3)
    qpos, kpos, _ = _positions()
    sliding_window = NO_WINDOW if tie else 1  # window=1 => early queries see no key
    ms = torch.zeros(B, H, Q, 1, requires_grad=True)  # finite sink seed
    dn = torch.ones(B, H, Q, 1, requires_grad=True)
    nu = torch.zeros(B, H, Q, D, requires_grad=True)
    q = torch.randn(B, H, Q, D, requires_grad=True)
    src = torch.randn(B, H, KEY, D, requires_grad=True)
    if tie:
        with torch.no_grad():  # identical key rows => exact ties in the max
            src[:, :, 1] = src[:, :, 0]
            src[:, :, 2] = src[:, :, 0]

    outs = csa._StreamingSoftmaxDenseUpdateFn.apply(ms, dn, nu, q, src, qpos, kpos, SCALE, sliding_window)
    assert all(torch.isfinite(o).all() for o in outs)
    (outs[0].sum() + outs[1].sum() + outs[2].sum()).backward()
    for t in (q, src, ms, dn, nu):
        assert torch.isfinite(_grad(t)).all()


# --------------------------------------------------------------------------- #
# structural guards for the two regressions found in review
# --------------------------------------------------------------------------- #
def test_forward_chunk_returns_three_values():
    """part_max/exp_old are backward-only and must not be returned by forward."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(csa._streaming_softmax_forward_chunk)))
    ret = max(
        len(n.value.elts)
        for n in ast.walk(tree)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
    )
    assert ret == 3


@pytest.mark.parametrize(
    "fn",
    [
        csa._StreamingSoftmaxDenseUpdateFn,
        csa._StreamingSoftmaxCompressedUpdateFn,
        csa._StreamingSoftmaxCompressedIndexedUpdateFn,
    ],
)
def test_backward_return_matches_forward_inputs(fn):
    n_forward_inputs = len(inspect.signature(fn.forward).parameters) - 1  # minus ctx
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn.backward)))
    n_backward_returns = max(
        len(n.value.elts)
        for n in ast.walk(tree)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
    )
    assert n_backward_returns == n_forward_inputs
