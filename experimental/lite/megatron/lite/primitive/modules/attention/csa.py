# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import math
from typing import Any

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from megatron.lite.primitive.modules.attention.dsa import rotate_activation
from megatron.lite.primitive.modules.attention.cp import (
    compress_contiguous_chunks_for_cp,
    iter_cp_sources,
)
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.primitive.utils.rotary import (
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
)


def _streaming_q_chunk(q_len: int, denom_elems: int) -> int:
    # Cap fp32 score materialization to ~256MB per q-chunk to bound peak memory.
    target_fp32_elements = 64 * 1024 * 1024
    return max(1, min(q_len, target_fp32_elements // max(1, denom_elems)))


def _streaming_softmax_forward_chunk(
    scores: torch.Tensor,
    values: torch.Tensor,
    max_chunk: torch.Tensor,
    denom_chunk: torch.Tensor,
    numer_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One online-softmax update; scores must already be masked (-inf on invalid)."""
    part_max = scores.max(dim=-1, keepdim=True).values
    new_max = torch.maximum(max_chunk, part_max)
    exp_old = torch.exp(max_chunk - new_max)
    exp_part = torch.exp(scores - new_max)
    new_denom = denom_chunk * exp_old + exp_part.sum(dim=-1, keepdim=True)
    new_numer = numer_chunk * exp_old + torch.matmul(exp_part, values)
    return new_max, new_denom, new_numer


def _streaming_softmax_backward_chunk(
    scores: torch.Tensor,
    values: torch.Tensor,
    max_chunk: torch.Tensor,
    denom_chunk: torch.Tensor,
    numer_chunk: torch.Tensor,
    grad_new_max_chunk: torch.Tensor,
    grad_new_denom_chunk: torch.Tensor,
    grad_new_numer_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VJP of one online-softmax update. Returns grads for scores and the value
    tensor plus the running-state grads; the caller maps grad_scores back to its own q/k."""
    part_max = scores.max(dim=-1, keepdim=True).values
    new_max = torch.maximum(max_chunk, part_max)
    exp_old = torch.exp(max_chunk - new_max)

    grad_denom_chunk = grad_new_denom_chunk * exp_old
    grad_numer_chunk = grad_new_numer_chunk * exp_old

    grad_exp_old = grad_new_denom_chunk * denom_chunk + (
        grad_new_numer_chunk * numer_chunk
    ).sum(dim=-1, keepdim=True)
    grad_exp_part = grad_new_denom_chunk + torch.matmul(grad_new_numer_chunk, values.transpose(-1, -2))
    exp_part = torch.exp(scores - new_max)
    grad_value = torch.matmul(exp_part.transpose(-1, -2), grad_new_numer_chunk)

    grad_scores = grad_exp_part * exp_part
    grad_new_max_from_exp_part = -(grad_scores.sum(dim=-1, keepdim=True))
    grad_new_max_total = grad_new_max_chunk + grad_new_max_from_exp_part - grad_exp_old * exp_old

    choose_part = part_max > max_chunk
    grad_max_scores_chunk = grad_exp_old * exp_old + grad_new_max_total * (
        ~choose_part
    ).to(max_chunk.dtype)

    part_mask = scores == part_max
    part_count = part_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(scores.dtype)
    grad_scores = grad_scores + (
        grad_new_max_total * choose_part.to(scores.dtype) / part_count
    ) * part_mask.to(scores.dtype)
    return grad_scores, grad_value, grad_max_scores_chunk, grad_denom_chunk, grad_numer_chunk


def _recompute_dense_scores(
    q_chunk_f: torch.Tensor,
    source_t: torch.Tensor,
    q_positions_chunk: torch.Tensor,
    k_positions: torch.Tensor,
    scale: float,
    sliding_window: int,
) -> torch.Tensor:
    scores = torch.matmul(q_chunk_f, source_t) * scale
    mask = _source_scores_mask(
        q_positions_chunk, k_positions, sliding_window=sliding_window
    ).unsqueeze(1)
    return scores.masked_fill(~mask, -float("inf"))


def _recompute_compressed_scores(
    q_chunk_f: torch.Tensor,
    comp_t: torch.Tensor,
    q_positions_chunk: torch.Tensor,
    comp_positions: torch.Tensor,
    ratio: int,
    scale: float,
) -> torch.Tensor:
    scores = torch.matmul(q_chunk_f, comp_t) * scale
    valid = _compressed_scores_mask(q_positions_chunk, comp_positions, ratio=ratio).unsqueeze(1)
    return scores.masked_fill(~valid, -float("inf"))


def _recompute_index_chunk(
    q_idx_chunk_f: torch.Tensor,
    k_idx_f: torch.Tensor,
    weights_chunk_f: torch.Tensor,
    q_positions_chunk: torch.Tensor,
    comp_positions: torch.Tensor,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = _compressed_scores_mask(q_positions_chunk, comp_positions, ratio=ratio)
    z = torch.einsum("bhsd,btd->bsht", q_idx_chunk_f, k_idx_f)
    relu_z = z.relu()
    index_chunk = (relu_z * weights_chunk_f.unsqueeze(-1)).sum(dim=2)
    index_chunk = index_chunk.masked_fill(~valid, -float("inf"))
    return valid, z, relu_z, index_chunk


class _StreamingSoftmaxDenseUpdateFn(torch.autograd.Function):
    """
    Dense (CP source) update with backward recompute: avoids caching per-source score tensors.
    """
    @staticmethod
    def forward(
        ctx,
        max_scores: torch.Tensor,
        denom: torch.Tensor,
        numer: torch.Tensor,
        q: torch.Tensor,
        source_heads: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
        scale: float,
        sliding_window: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_f = q.float()
        source_f = source_heads.float()
        source_t = source_f.transpose(-1, -2)
        batch, n_heads, q_len, _ = q_f.shape
        key_len = source_f.size(-2)
        q_chunk = _streaming_q_chunk(q_len, batch * n_heads * key_len)

        new_max_chunks = []
        new_denom_chunks = []
        new_numer_chunks = []
        for start in range(0, q_len, q_chunk):
            end = min(start + q_chunk, q_len)
            scores = _recompute_dense_scores(
                q_f[..., start:end, :], source_t, q_positions[:, start:end],
                k_positions, scale, sliding_window,
            )
            new_max, new_denom, new_numer = _streaming_softmax_forward_chunk(
                scores, source_f,
                max_scores[..., start:end, :],
                denom[..., start:end, :],
                numer[..., start:end, :],
            )
            new_max_chunks.append(new_max)
            new_denom_chunks.append(new_denom)
            new_numer_chunks.append(new_numer)

        new_max = torch.cat(new_max_chunks, dim=2)
        new_denom = torch.cat(new_denom_chunks, dim=2)
        new_numer = torch.cat(new_numer_chunks, dim=2)

        ctx.scale = scale
        ctx.sliding_window = sliding_window
        ctx.q_chunk = q_chunk
        ctx.save_for_backward(
            max_scores,
            denom,
            numer,
            q,
            source_heads,
            q_positions,
            k_positions,
        )
        return new_max, new_denom, new_numer

    @staticmethod
    def backward(
        ctx,
        grad_new_max: torch.Tensor,
        grad_new_denom: torch.Tensor,
        grad_new_numer: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
    ]:
        (
            max_scores,
            denom,
            numer,
            q,
            source_heads,
            q_positions,
            k_positions,
        ) = ctx.saved_tensors
        scale = ctx.scale
        sliding_window = ctx.sliding_window
        q_chunk = ctx.q_chunk

        q_f = q.float()
        source_f = source_heads.float()
        source_t = source_f.transpose(-1, -2)
        q_len = q_f.size(-2)
        grad_max_scores = torch.empty_like(max_scores)
        grad_denom = torch.empty_like(denom)
        grad_numer = torch.empty_like(numer)
        grad_q = torch.empty_like(q_f)
        grad_source_heads = torch.zeros_like(source_f)

        for start in range(0, q_len, q_chunk):
            end = min(start + q_chunk, q_len)
            scores = _recompute_dense_scores(
                q_f[..., start:end, :], source_t, q_positions[:, start:end],
                k_positions, scale, sliding_window,
            )
            grad_scores, grad_value, grad_max_scores_chunk, grad_denom_chunk, grad_numer_chunk = (
                _streaming_softmax_backward_chunk(
                    scores, source_f,
                    max_scores[..., start:end, :],
                    denom[..., start:end, :],
                    numer[..., start:end, :],
                    grad_new_max[..., start:end, :],
                    grad_new_denom[..., start:end, :],
                    grad_new_numer[..., start:end, :],
                )
            )

            grad_q_chunk = torch.matmul(grad_scores, source_f) * scale
            grad_source_heads = grad_source_heads + (
                torch.matmul(grad_scores.transpose(-1, -2), q_f[..., start:end, :]) * scale
                + grad_value
            )

            grad_max_scores[..., start:end, :] = grad_max_scores_chunk
            grad_denom[..., start:end, :] = grad_denom_chunk
            grad_numer[..., start:end, :] = grad_numer_chunk
            grad_q[..., start:end, :] = grad_q_chunk

        return (
            grad_max_scores,
            grad_denom,
            grad_numer,
            grad_q.to(dtype=q.dtype),
            grad_source_heads.to(dtype=source_heads.dtype),
            None,
            None,
            None,
            None,
        )


class _StreamingSoftmaxCompressedUpdateFn(torch.autograd.Function):
    """
    Compressed update with backward recompute: avoids caching compressed score tensors.
    """
    @staticmethod
    def forward(
        ctx,
        max_scores: torch.Tensor,
        denom: torch.Tensor,
        numer: torch.Tensor,
        q: torch.Tensor,
        compressed_values: torch.Tensor,
        q_positions: torch.Tensor,
        comp_positions: torch.Tensor,
        ratio: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_f = q.float()
        comp_f = compressed_values.float()
        comp_t = comp_f.transpose(-1, -2)
        batch, n_heads, q_len, _ = q_f.shape
        key_len = comp_f.size(-2)
        q_chunk = _streaming_q_chunk(q_len, batch * n_heads * key_len)

        new_max_chunks = []
        new_denom_chunks = []
        new_numer_chunks = []
        for start in range(0, q_len, q_chunk):
            end = min(start + q_chunk, q_len)
            scores = _recompute_compressed_scores(
                q_f[..., start:end, :], comp_t, q_positions[:, start:end],
                comp_positions, ratio, scale,
            )
            new_max, new_denom, new_numer = _streaming_softmax_forward_chunk(
                scores, comp_f,
                max_scores[..., start:end, :],
                denom[..., start:end, :],
                numer[..., start:end, :],
            )
            new_max_chunks.append(new_max)
            new_denom_chunks.append(new_denom)
            new_numer_chunks.append(new_numer)

        new_max = torch.cat(new_max_chunks, dim=2)
        new_denom = torch.cat(new_denom_chunks, dim=2)
        new_numer = torch.cat(new_numer_chunks, dim=2)

        ctx.scale = scale
        ctx.ratio = ratio
        ctx.q_chunk = q_chunk
        ctx.save_for_backward(
            max_scores,
            denom,
            numer,
            q,
            compressed_values,
            q_positions,
            comp_positions,
        )
        return new_max, new_denom, new_numer

    @staticmethod
    def backward(
        ctx,
        grad_new_max: torch.Tensor,
        grad_new_denom: torch.Tensor,
        grad_new_numer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        (
            max_scores,
            denom,
            numer,
            q,
            compressed_values,
            q_positions,
            comp_positions,
        ) = ctx.saved_tensors
        scale = ctx.scale
        ratio = ctx.ratio
        q_chunk = ctx.q_chunk

        q_f = q.float()
        comp_f = compressed_values.float()
        comp_t = comp_f.transpose(-1, -2)
        q_len = q_f.size(-2)
        grad_max_scores = torch.empty_like(max_scores)
        grad_denom = torch.empty_like(denom)
        grad_numer = torch.empty_like(numer)
        grad_q = torch.empty_like(q_f)
        grad_comp = torch.zeros_like(comp_f)

        for start in range(0, q_len, q_chunk):
            end = min(start + q_chunk, q_len)
            scores = _recompute_compressed_scores(
                q_f[..., start:end, :], comp_t, q_positions[:, start:end],
                comp_positions, ratio, scale,
            )
            grad_scores, grad_value, grad_max_scores_chunk, grad_denom_chunk, grad_numer_chunk = (
                _streaming_softmax_backward_chunk(
                    scores, comp_f,
                    max_scores[..., start:end, :],
                    denom[..., start:end, :],
                    numer[..., start:end, :],
                    grad_new_max[..., start:end, :],
                    grad_new_denom[..., start:end, :],
                    grad_new_numer[..., start:end, :],
                )
            )

            grad_q_chunk = torch.matmul(grad_scores, comp_f) * scale
            grad_comp = grad_comp + (
                torch.matmul(grad_scores.transpose(-1, -2), q_f[..., start:end, :]) * scale
                + grad_value
            )

            grad_max_scores[..., start:end, :] = grad_max_scores_chunk
            grad_denom[..., start:end, :] = grad_denom_chunk
            grad_numer[..., start:end, :] = grad_numer_chunk
            grad_q[..., start:end, :] = grad_q_chunk

        return (
            grad_max_scores,
            grad_denom,
            grad_numer,
            grad_q.to(dtype=q.dtype),
            grad_comp.to(dtype=compressed_values.dtype),
            None,
            None,
            None,
            None,
        )


class _StreamingSoftmaxCompressedIndexedUpdateFn(torch.autograd.Function):
    """
    Compressed + indexer update with backward recompute: avoids caching index/compressed scores.
    """
    @staticmethod
    def forward(
        ctx,
        max_scores: torch.Tensor,
        denom: torch.Tensor,
        numer: torch.Tensor,
        q: torch.Tensor,
        compressed_values: torch.Tensor,
        q_positions: torch.Tensor,
        comp_positions: torch.Tensor,
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
        index_weights: torch.Tensor,
        index_topk: int,
        ratio: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_f = q.float()
        comp_f = compressed_values.float()
        comp_t = comp_f.transpose(-1, -2)
        q_idx_f = q_idx.float()
        k_idx_f = k_idx.float()
        weights_f = index_weights.float()

        batch, n_heads, q_len, _ = q_f.shape
        key_len = comp_f.size(-2)
        index_heads = q_idx_f.size(1)
        q_chunk = _streaming_q_chunk(q_len, batch * max(n_heads, index_heads) * key_len)
        topk = min(index_topk, key_len)

        new_max_chunks = []
        new_denom_chunks = []
        new_numer_chunks = []
        topk_chunks = []
        for start in range(0, q_len, q_chunk):
            end = min(start + q_chunk, q_len)
            scores = torch.matmul(q_f[..., start:end, :], comp_t) * scale
            valid, _z, _relu_z, index_chunk = _recompute_index_chunk(
                q_idx_f[:, :, start:end, :], k_idx_f, weights_f[:, start:end, :],
                q_positions[:, start:end], comp_positions, ratio,
            )

            topk_idx = index_chunk.topk(topk, dim=-1).indices
            topk_mask = torch.zeros_like(index_chunk, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)
            scores = (scores + index_chunk.unsqueeze(1)).masked_fill(
                ~(valid & topk_mask).unsqueeze(1),
                -float("inf"),
            )

            new_max, new_denom, new_numer = _streaming_softmax_forward_chunk(
                scores, comp_f,
                max_scores[..., start:end, :],
                denom[..., start:end, :],
                numer[..., start:end, :],
            )
            new_max_chunks.append(new_max)
            new_denom_chunks.append(new_denom)
            new_numer_chunks.append(new_numer)
            topk_chunks.append(topk_idx)

        new_max = torch.cat(new_max_chunks, dim=2)
        new_denom = torch.cat(new_denom_chunks, dim=2)
        new_numer = torch.cat(new_numer_chunks, dim=2)
        topk_indices = torch.cat(topk_chunks, dim=1)

        ctx.scale = scale
        ctx.ratio = ratio
        ctx.q_chunk = q_chunk
        ctx.topk = topk
        ctx.save_for_backward(
            max_scores,
            denom,
            numer,
            q,
            compressed_values,
            q_positions,
            comp_positions,
            q_idx,
            k_idx,
            index_weights,
            topk_indices,
        )
        return new_max, new_denom, new_numer

    @staticmethod
    def backward(
        ctx,
        grad_new_max: torch.Tensor,
        grad_new_denom: torch.Tensor,
        grad_new_numer: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
    ]:
        (
            max_scores,
            denom,
            numer,
            q,
            compressed_values,
            q_positions,
            comp_positions,
            q_idx,
            k_idx,
            index_weights,
            topk_indices,
        ) = ctx.saved_tensors
        scale = ctx.scale
        ratio = ctx.ratio
        q_chunk = ctx.q_chunk
        topk = ctx.topk

        q_f = q.float()
        comp_f = compressed_values.float()
        comp_t = comp_f.transpose(-1, -2)
        q_idx_f = q_idx.float()
        k_idx_f = k_idx.float()
        weights_f = index_weights.float()

        q_len = q_f.size(-2)
        grad_max_scores = torch.empty_like(max_scores)
        grad_denom = torch.empty_like(denom)
        grad_numer = torch.empty_like(numer)
        grad_q = torch.empty_like(q_f)
        grad_comp = torch.zeros_like(comp_f)
        grad_q_idx = torch.empty_like(q_idx_f)
        grad_k_idx = torch.zeros_like(k_idx_f)
        grad_index_weights = torch.empty_like(weights_f)

        for start in range(0, q_len, q_chunk):
            end = min(start + q_chunk, q_len)
            scores_base = torch.matmul(q_f[..., start:end, :], comp_t) * scale
            valid, z, relu_z, index_chunk = _recompute_index_chunk(
                q_idx_f[:, :, start:end, :], k_idx_f, weights_f[:, start:end, :],
                q_positions[:, start:end], comp_positions, ratio,
            )

            topk_idx = topk_indices[:, start:end, :]
            topk_mask = torch.zeros_like(index_chunk, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)
            final_mask = valid & topk_mask
            scores = (scores_base + index_chunk.unsqueeze(1)).masked_fill(
                ~final_mask.unsqueeze(1),
                -float("inf"),
            )

            grad_scores, grad_value, grad_max_scores_chunk, grad_denom_chunk, grad_numer_chunk = (
                _streaming_softmax_backward_chunk(
                    scores, comp_f,
                    max_scores[..., start:end, :],
                    denom[..., start:end, :],
                    numer[..., start:end, :],
                    grad_new_max[..., start:end, :],
                    grad_new_denom[..., start:end, :],
                    grad_new_numer[..., start:end, :],
                )
            )

            grad_q_chunk = torch.matmul(grad_scores, comp_f) * scale
            grad_comp = grad_comp + (
                torch.matmul(grad_scores.transpose(-1, -2), q_f[..., start:end, :]) * scale
                + grad_value
            )

            grad_index = grad_scores.sum(dim=1)
            grad_weights_chunk = (grad_index.unsqueeze(2) * relu_z).sum(dim=-1)
            grad_relu = grad_index.unsqueeze(2) * weights_f[:, start:end, :].unsqueeze(-1)
            grad_z = grad_relu * (z > 0).to(z.dtype)
            grad_q_idx_chunk = torch.einsum("bsht,btd->bhsd", grad_z, k_idx_f)
            grad_k_idx = grad_k_idx + torch.einsum(
                "bsht,bhsd->btd", grad_z, q_idx_f[:, :, start:end, :]
            )

            grad_max_scores[..., start:end, :] = grad_max_scores_chunk
            grad_denom[..., start:end, :] = grad_denom_chunk
            grad_numer[..., start:end, :] = grad_numer_chunk
            grad_q[..., start:end, :] = grad_q_chunk
            grad_q_idx[:, :, start:end, :] = grad_q_idx_chunk
            grad_index_weights[:, start:end, :] = grad_weights_chunk

        return (
            grad_max_scores,
            grad_denom,
            grad_numer,
            grad_q.to(dtype=q.dtype),
            grad_comp.to(dtype=compressed_values.dtype),
            None,
            None,
            grad_q_idx.to(dtype=q_idx.dtype),
            grad_k_idx.to(dtype=k_idx.dtype),
            grad_index_weights.to(dtype=index_weights.dtype),
            None,  # index_topk
            None,  # ratio
            None,  # scale
        )


class GroupedLinear(nn.Module):
    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int):
        super().__init__()
        self.in_features_per_group = in_features_per_group
        self.out_features = out_features
        self.n_groups = n_groups
        self.weight = nn.Parameter(torch.empty(out_features, in_features_per_group))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_per_group = self.out_features // self.n_groups
        weight = self.weight.view(self.n_groups, out_per_group, self.in_features_per_group)
        return torch.einsum("...gd,god->...go", x, weight)


def build_rope_cos_sin(
    position_ids: torch.Tensor,
    rope_head_dim: int,
    rope_theta: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_head_dim, 2, device=device, dtype=torch.float32) / rope_head_dim)
    )
    freqs = torch.einsum("bs,d->bsd", position_ids.to(torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def build_yarn_rope_cos_sin(
    position_ids: torch.Tensor,
    rope_head_dim: int,
    rope_theta: float,
    *,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = rope_head_dim
    inv_freq_extra = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    inv_freq_inter = 1.0 / (
        config.rotary_scaling_factor
        * rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    low, high = _yarn_find_correction_range(
        config.beta_fast,
        config.beta_slow,
        dim,
        rope_theta,
        config.original_max_position_embeddings,
    )
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, device)
    inv_freq = inv_freq_inter * (1 - inv_freq_mask) + inv_freq_extra * inv_freq_mask
    freqs = torch.einsum("bs,d->bsd", position_ids.to(torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def build_compressed_rope_cos_sin(
    position_ids: torch.Tensor,
    rope_head_dim: int,
    rope_theta: float,
    *,
    config: Any,
    use_yarn: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_yarn:
        return build_yarn_rope_cos_sin(
            position_ids,
            rope_head_dim,
            rope_theta,
            config=config,
            device=device,
            dtype=dtype,
        )
    return build_rope_cos_sin(position_ids, rope_head_dim, rope_theta, device=device, dtype=dtype)


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int
) -> torch.Tensor:
    if rope_head_dim == 0:
        return x
    rope = x[..., -rope_head_dim:]
    tail = x[..., :-rope_head_dim]
    rope_pairs = rope.unflatten(-1, (-1, 2))
    a, b = rope_pairs[..., 0], rope_pairs[..., 1]
    c = cos[..., : rope_head_dim // 2]
    s = sin[..., : rope_head_dim // 2]
    while c.ndim < a.ndim:
        c = c.unsqueeze(1)
        s = s.unsqueeze(1)
    out_a = a * c - b * s
    out_b = a * s + b * c
    rope_out = torch.stack([out_a, out_b], dim=-1).flatten(-2)
    return torch.cat([tail, rope_out], dim=-1)


class CompressedSequenceCompressor(nn.Module):
    def __init__(self, config: Any, compress_ratio: int, head_dim: int, *, rotate: bool = False):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = min(config.qk_rope_head_dim, head_dim)
        self.overlap = compress_ratio == 4
        self.coff = 2 if self.overlap else 1
        self.rotate = rotate
        self.wkv = nn.Linear(config.hidden_size, self.coff * head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.coff * head_dim, bias=False)
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, self.coff * head_dim, dtype=torch.float32)
        )
        self.norm = te.RMSNorm(head_dim, eps=config.rms_norm_eps)
        nn.init.normal_(self.ape, mean=0.0, std=config.initializer_range)

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
        bsz, n_blocks, ratio, _, head_dim = tensor.shape
        out = tensor.new_full((bsz, n_blocks, 2 * ratio, head_dim), fill_value)
        out[:, :, ratio:] = tensor[:, :, :, 1]
        out[:, 1:, :ratio] = tensor[:, :-1, :, 0]
        return out

    def forward(
        self, x: torch.Tensor, *, position_ids: torch.Tensor, rope_theta: float
    ) -> torch.Tensor | None:
        bsz, seq_len, _ = x.shape
        ratio = self.compress_ratio
        n_blocks = seq_len // ratio
        if n_blocks == 0:
            return None
        cutoff = n_blocks * ratio
        content = self.wkv(x[:, :cutoff])
        gate = self.wgate(x[:, :cutoff])
        content = content.view(bsz, n_blocks, ratio, self.coff, self.head_dim)
        gate = gate.view_as(content)
        gate = gate + self.ape.view(1, 1, ratio, self.coff, self.head_dim).to(gate.device)
        if self.overlap:
            content = self._overlap_transform(content, 0.0)
            gate = self._overlap_transform(gate, float("-inf"))
        else:
            content = content.squeeze(3)
            gate = gate.squeeze(3)
        weights = torch.softmax(gate.float(), dim=2).to(dtype=content.dtype)
        compressed = self.norm((content * weights).sum(dim=2)).unsqueeze(1)
        compressed_positions = position_ids[:, :cutoff:ratio]
        cos, sin = build_compressed_rope_cos_sin(
            compressed_positions,
            self.rope_head_dim,
            rope_theta,
            config=self.config,
            use_yarn=self.compress_ratio > 1,
            device=x.device,
            dtype=compressed.dtype,
        )
        compressed = apply_partial_rope(compressed, cos, sin, self.rope_head_dim)
        return rotate_activation(compressed) if self.rotate else compressed


def _source_scores_mask(
    q_positions: torch.Tensor, k_positions: torch.Tensor, *, sliding_window: int
) -> torch.Tensor:
    q_pos = q_positions.unsqueeze(-1)
    k_pos = k_positions.unsqueeze(1)
    return (k_pos <= q_pos) & (k_pos >= q_pos - sliding_window + 1)


def _compressed_scores_mask(
    q_positions: torch.Tensor, comp_positions: torch.Tensor, *, ratio: int
) -> torch.Tensor:
    visible = (q_positions + 1) // ratio
    comp_ids = comp_positions // ratio
    return comp_ids.unsqueeze(1) < visible.unsqueeze(-1)


def _window_topk_indices(
    batch: int, seq_len: int, window: int, *, device: torch.device
) -> torch.Tensor:
    topk = max(1, min(int(window), seq_len))
    query_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
    offsets = torch.arange(topk, device=device)
    indices = (query_pos - topk + 1).clamp(min=0) + offsets
    indices = torch.where(indices > query_pos, -1, indices)
    return indices.unsqueeze(0).expand(batch, -1, -1).to(torch.int32)


def _load_dsa_kernels():
    from megatron.lite.primitive.kernels import dsa_kernels

    return dsa_kernels


class CompressedSparseAttentionIndexer(nn.Module):
    def __init__(self, config, compress_ratio: int):
        super().__init__()
        self.config = config
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.rope_head_dim = min(config.qk_rope_head_dim, config.index_head_dim)
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = nn.Linear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        self.compressor = CompressedSequenceCompressor(
            config, compress_ratio, config.index_head_dim, rotate=True
        )


class CompressedSparseAttention(nn.Module):
    def __init__(
        self,
        config,
        *,
        layer_idx: int,
        ps: ParallelState,
    ):
        super().__init__()
        self.config = config
        self.ps = ps
        self.attention_backend = "torch"
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.num_heads_per_group = config.num_attention_heads // config.o_groups
        # MTP layers use layer_idx == num_hidden_layers (+i), which is past the
        # per-decoder-layer compress_ratios list (length num_hidden_layers); fall
        # back to the last real layer's ratio so the MTP CSA still builds.
        if config.compress_ratios:
            _cr_idx = min(layer_idx, len(config.compress_ratios) - 1)
            self.compress_ratio = config.compress_ratios[_cr_idx]
        else:
            self.compress_ratio = 0
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = te.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = te.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.wo_a = GroupedLinear(
            self.num_heads_per_group * self.head_dim,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.compressor = (
            CompressedSequenceCompressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio > 1
            else None
        )
        self.indexer = (
            CompressedSparseAttentionIndexer(config, self.compress_ratio)
            if self.compress_ratio == 4
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.ps.cp_size > 1 and attention_mask is not None:
            raise ValueError("CP expects attention_mask=None; masks are derived from position_ids.")
        batch, seq_len, _ = x.shape
        attention_rope_theta = (
            self.config.compress_rope_theta if self.compress_ratio > 1 else self.config.rope_theta
        )
        cos, sin = build_compressed_rope_cos_sin(
            position_ids,
            self.rope_head_dim,
            attention_rope_theta,
            config=self.config,
            use_yarn=self.compress_ratio > 1,
            device=x.device,
            dtype=x.dtype,
        )
        q_low = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_low).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(
            q.float().pow(2).mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
        ).to(dtype=q.dtype)
        kv = self.kv_norm(self.wkv(x)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        q = apply_partial_rope(q, cos, sin, self.rope_head_dim)
        kv = apply_partial_rope(kv, cos, sin, self.rope_head_dim)
        use_sparse_backend = self.attention_backend not in {"local", "eager", "torch"}
        if (
            use_sparse_backend
            and self.compress_ratio == 4
            and self.ps.cp_size == 1
            and self.compressor is not None
            and self.indexer is not None
        ):
            return self._forward_fused_dsa_cp1(
                x,
                q,
                q_low,
                kv,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
            )
        if use_sparse_backend and self.ps.cp_size == 1 and attention_mask is None:
            return self._forward_fused_sparse_no_indexer_cp1(
                x,
                q,
                kv,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
            )

        sink = self.sinks.view(1, self.num_heads, 1, 1).expand(batch, -1, seq_len, -1)
        if attention_mask is not None:
            dense_score_parts = []
            dense_value_parts = []
            for _source_rank, source_kv, source_pos in iter_cp_sources(
                kv,
                position_ids,
                cp_rank=self.ps.cp_rank,
                cp_size=self.ps.cp_size,
                cp_group=self.ps.cp_group,
            ):
                source_heads = source_kv.expand(-1, self.num_heads, -1, -1)
                scores = torch.matmul(q.float(), source_heads.float().transpose(-1, -2)) / (
                    self.head_dim**0.5
                )
                source_mask = _source_scores_mask(
                    position_ids,
                    source_pos,
                    sliding_window=self.config.sliding_window,
                ).unsqueeze(1)
                scores = scores.masked_fill(~source_mask, -float("inf"))
                dense_score_parts.append(scores)
                dense_value_parts.append(source_heads)

            dense_scores = torch.cat(dense_score_parts, dim=-1)
            dense_scores = dense_scores + attention_mask.to(dtype=dense_scores.dtype)
            score_parts = [dense_scores]
            value_parts = [torch.cat(dense_value_parts, dim=2)]
        else:
            max_scores, denom, numer = self._streaming_softmax_init(sink)
            for _source_rank, source_kv, source_pos in iter_cp_sources(
                kv,
                position_ids,
                cp_rank=self.ps.cp_rank,
                cp_size=self.ps.cp_size,
                cp_group=self.ps.cp_group,
            ):
                source_heads = source_kv.expand(-1, self.num_heads, -1, -1)
                max_scores, denom, numer = self._streaming_softmax_update_dense_recompute(
                    max_scores,
                    denom,
                    numer,
                    q,
                    source_heads,
                    position_ids,
                    source_pos,
                )

        if self.compressor is not None:
            compressed_pack = compress_contiguous_chunks_for_cp(
                self.compressor,
                x,
                position_ids=position_ids,
                cp_rank=self.ps.cp_rank,
                cp_size=self.ps.cp_size,
                cp_group=self.ps.cp_group,
                compress_kwargs={"rope_theta": self.config.compress_rope_theta},
            )
        else:
            compressed_pack = None
        if compressed_pack is not None:
            compressed, compressed_pos = compressed_pack
            compressed, compressed_pos = self._gather_cp_sources(
                compressed, compressed_pos, seq_dim=2
            )
            compressed_values = compressed.expand(-1, self.num_heads, -1, -1)
            if attention_mask is not None:
                compressed_scores = torch.matmul(
                    q.float(), compressed_values.float().transpose(-1, -2)
                ) / (self.head_dim**0.5)
                compressed_valid = _compressed_scores_mask(
                    position_ids,
                    compressed_pos,
                    ratio=self.compress_ratio,
                ).unsqueeze(1)
                compressed_scores = compressed_scores.masked_fill(~compressed_valid, -float("inf"))

                if self.indexer is not None:
                    index_comp_pack = compress_contiguous_chunks_for_cp(
                        self.indexer.compressor,
                        x,
                        position_ids=position_ids,
                        cp_rank=self.ps.cp_rank,
                        cp_size=self.ps.cp_size,
                        cp_group=self.ps.cp_group,
                        compress_kwargs={"rope_theta": self.config.compress_rope_theta},
                    )
                    if index_comp_pack is not None:
                        index_comp, index_pos, q_idx, index_weights = (
                            self._build_index_query_weights(
                                index_comp_pack, x, q_low, position_ids, batch, seq_len
                            )
                        )
                        k_idx = index_comp.squeeze(1)
                        index_scores = torch.einsum(
                            "bhsd,btd->bsht", q_idx.float(), k_idx.float()
                        ).relu()
                        index_scores = (index_scores * index_weights.unsqueeze(-1)).sum(dim=2)
                        index_valid = _compressed_scores_mask(
                            position_ids,
                            index_pos,
                            ratio=self.compress_ratio,
                        )
                        index_scores = index_scores.masked_fill(~index_valid, -float("inf"))
                        topk = min(self.indexer.index_topk, index_scores.size(-1))
                        topk_indices = index_scores.topk(topk, dim=-1).indices
                        topk_mask = torch.zeros_like(index_scores, dtype=torch.bool)
                        topk_mask.scatter_(-1, topk_indices, True)
                        compressed_scores = compressed_scores + index_scores.unsqueeze(1)
                        compressed_scores = compressed_scores.masked_fill(
                            ~topk_mask.unsqueeze(1), -float("inf")
                        )

                score_parts.append(compressed_scores)
                value_parts.append(compressed_values)
            else:
                if self.indexer is None:
                    max_scores, denom, numer = self._streaming_softmax_update_compressed_recompute(
                        max_scores,
                        denom,
                        numer,
                        q,
                        compressed_values,
                        position_ids,
                        compressed_pos,
                    )
                else:
                    index_comp_pack = compress_contiguous_chunks_for_cp(
                        self.indexer.compressor,
                        x,
                        position_ids=position_ids,
                        cp_rank=self.ps.cp_rank,
                        cp_size=self.ps.cp_size,
                        cp_group=self.ps.cp_group,
                        compress_kwargs={"rope_theta": self.config.compress_rope_theta},
                    )
                    if index_comp_pack is not None:
                        index_comp, index_pos, q_idx, index_weights = (
                            self._build_index_query_weights(
                                index_comp_pack, x, q_low, position_ids, batch, seq_len
                            )
                        )
                        # The streaming indexed Function builds the index mask from
                        # compressed_pos (not index_pos); this holds only while the indexer
                        # shares the main compressor's ratio/chunking. Guard against drift.
                        assert torch.equal(index_pos, compressed_pos), (
                            "indexer positions diverged from compressor positions; "
                            "streaming index mask would be wrong"
                        )
                        max_scores, denom, numer = (
                            self._streaming_softmax_update_compressed_indexed_recompute(
                                max_scores,
                                denom,
                                numer,
                                q,
                                compressed_values,
                                position_ids,
                                compressed_pos,
                                q_idx,
                                index_comp.squeeze(1),
                                index_weights,
                                int(self.indexer.index_topk),
                            )
                        )
                    else:
                        max_scores, denom, numer = self._streaming_softmax_update_compressed_recompute(
                            max_scores,
                            denom,
                            numer,
                            q,
                            compressed_values,
                            position_ids,
                            compressed_pos,
                        )

        if attention_mask is not None:
            context = self._streaming_softmax_context(score_parts, value_parts, sink, q.dtype)
        else:
            context = self._streaming_softmax_finalize(denom, numer, q.dtype)

        return self._project_context(context, cos, sin)

    def _build_index_query_weights(
        self,
        index_comp_pack: tuple[torch.Tensor, torch.Tensor],
        x: torch.Tensor,
        q_low: torch.Tensor,
        position_ids: torch.Tensor,
        batch: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared indexer path: CP-gathered index KV/positions + RoPE'd index query and
        scaled index weights.

        Used by both the eager (attention-mask) and streaming (CP) indexed paths so
        the two cannot drift."""
        assert self.indexer is not None
        index_comp, index_pos = index_comp_pack
        index_comp, index_pos = self._gather_cp_sources(index_comp, index_pos, seq_dim=2)
        idx_cos, idx_sin = build_compressed_rope_cos_sin(
            position_ids,
            self.indexer.rope_head_dim,
            self.config.compress_rope_theta,
            config=self.config,
            use_yarn=self.compress_ratio > 1,
            device=x.device,
            dtype=x.dtype,
        )
        q_idx = (
            self.indexer.wq_b(q_low)
            .view(batch, seq_len, self.indexer.index_n_heads, self.indexer.index_head_dim)
            .transpose(1, 2)
        )
        q_idx = apply_partial_rope(q_idx, idx_cos, idx_sin, self.indexer.rope_head_dim)
        index_weights = (
            self.indexer.weights_proj(x).float()
            * (self.indexer.index_n_heads**-0.5)
            * self.indexer.softmax_scale
        )
        return index_comp, index_pos, q_idx, index_weights

    def _streaming_softmax_init(
        self,
        sink_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_heads, q_len, _ = sink_scores.shape
        # Start from sink term (finite) to avoid -inf - (-inf) when a score block is fully masked.
        max_scores = sink_scores.float()
        denom = torch.ones(
            (batch, n_heads, q_len, 1),
            device=sink_scores.device,
            dtype=torch.float32,
        )
        numer = torch.zeros(
            (batch, n_heads, q_len, self.head_dim),
            device=sink_scores.device,
            dtype=torch.float32,
        )
        return max_scores, denom, numer

    def _streaming_softmax_update_dense_recompute(
        self,
        max_scores: torch.Tensor,
        denom: torch.Tensor,
        numer: torch.Tensor,
        q: torch.Tensor,
        source_heads: torch.Tensor,
        position_ids: torch.Tensor,
        source_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _StreamingSoftmaxDenseUpdateFn.apply(
            max_scores,
            denom,
            numer,
            q,
            source_heads,
            position_ids,
            source_pos,
            self.head_dim**-0.5,
            int(self.config.sliding_window),
        )

    def _streaming_softmax_update_compressed_recompute(
        self,
        max_scores: torch.Tensor,
        denom: torch.Tensor,
        numer: torch.Tensor,
        q: torch.Tensor,
        compressed_values: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _StreamingSoftmaxCompressedUpdateFn.apply(
            max_scores,
            denom,
            numer,
            q,
            compressed_values,
            position_ids,
            compressed_pos,
            int(self.compress_ratio),
            self.head_dim**-0.5,
        )

    def _streaming_softmax_update_compressed_indexed_recompute(
        self,
        max_scores: torch.Tensor,
        denom: torch.Tensor,
        numer: torch.Tensor,
        q: torch.Tensor,
        compressed_values: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_pos: torch.Tensor,
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
        index_weights: torch.Tensor,
        index_topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _StreamingSoftmaxCompressedIndexedUpdateFn.apply(
            max_scores,
            denom,
            numer,
            q,
            compressed_values,
            position_ids,
            compressed_pos,
            q_idx,
            k_idx,
            index_weights,
            index_topk,
            int(self.compress_ratio),
            self.head_dim**-0.5,
        )

    def _streaming_softmax_finalize(
        self,
        denom: torch.Tensor,
        numer: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return (numer / denom).to(dtype=out_dtype)

    def _streaming_softmax_context(
        self,
        score_parts: list[torch.Tensor],
        value_parts: list[torch.Tensor],
        sink_scores: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Attention-mask path: eager streaming update is concise and numerically identical.
        max_scores, denom, numer = self._streaming_softmax_init(sink_scores)

        for scores, values in zip(score_parts, value_parts, strict=True):
            max_scores, denom, numer = _streaming_softmax_forward_chunk(
                scores.float(), values.float(), max_scores, denom, numer
            )

        return self._streaming_softmax_finalize(denom, numer, out_dtype)

    def _project_context(
        self, context: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        context = apply_partial_rope(context, cos, -sin, self.rope_head_dim).transpose(1, 2)
        batch, seq_len = context.shape[:2]
        grouped = context.reshape(
            batch, seq_len, self.config.o_groups, self.num_heads_per_group * self.head_dim
        )
        return self.wo_b(self.wo_a(grouped).flatten(2))

    def _gather_cp_sources(
        self, tensor: torch.Tensor, position_ids: torch.Tensor, *, seq_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts, pos_parts = [], []
        for _rank, source, source_pos in iter_cp_sources(
            tensor,
            position_ids,
            cp_rank=self.ps.cp_rank,
            cp_size=self.ps.cp_size,
            cp_group=self.ps.cp_group,
        ):
            parts.append(source)
            pos_parts.append(source_pos)
        pos = torch.cat(pos_parts, dim=1)
        return torch.cat(parts, dim=seq_dim), pos

    def _forward_fused_sparse_no_indexer_cp1(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        dsa_kernels = _load_dsa_kernels()
        batch, seq_len, _ = x.shape
        query = q.transpose(1, 2).transpose(0, 1).contiguous()
        kv_full = kv.squeeze(1)
        kv_full = kv_full.transpose(0, 1).contiguous()
        window_idxs = _window_topk_indices(
            batch,
            seq_len,
            self.config.sliding_window,
            device=x.device,
        )

        compressed = None
        if self.compressor is not None and self.compress_ratio > 1:
            compressed = self.compressor(
                x,
                position_ids=position_ids,
                rope_theta=self.config.compress_rope_theta,
            )
            if compressed is not None:
                compressed_kv = compressed.squeeze(1)
                kv_full = torch.cat([kv_full, compressed_kv.transpose(0, 1).contiguous()], dim=0)

        if compressed is not None:
            n_compressed = compressed.size(2)
            comp_idx = torch.arange(n_compressed, device=x.device).view(1, n_compressed)
            valid_per_pos = (
                torch.arange(1, seq_len + 1, device=x.device) // self.compress_ratio
            ).view(seq_len, 1)
            compress_topk_idxs = torch.where(
                comp_idx < valid_per_pos,
                comp_idx + seq_len,
                torch.full_like(comp_idx, -1),
            )
            compress_topk_idxs = (
                compress_topk_idxs.unsqueeze(0).expand(batch, -1, -1).to(torch.int32)
            )
            flat_idxs, _flat_tlen = dsa_kernels.build_flat_topk_idxs(
                window_idxs,
                compress_topk_idxs,
                batch_size=batch,
                seqlen_kv=kv_full.size(0),
            )
        else:
            flat_idxs, _flat_tlen = dsa_kernels.build_flat_topk_idxs(
                window_idxs,
                batch_size=batch,
                seqlen_kv=kv_full.size(0),
            )

        out = dsa_kernels.dsa_sparse_attn(
            query,
            kv_full,
            self.sinks.float(),
            flat_idxs,
            self.head_dim**-0.5,
        )
        context = (
            out.view(seq_len, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()
        )
        return self._project_context(context, cos, sin)

    def _forward_fused_dsa_cp1(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        q_low: torch.Tensor,
        kv: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.ps.cp_size != 1:
            raise NotImplementedError("DeepSeek V4 fused DSA path currently supports CP=1 only.")
        if attention_mask is not None:
            raise NotImplementedError(
                "DeepSeek V4 fused DSA path currently supports causal masking only."
            )
        dsa_kernels = _load_dsa_kernels()
        # The cuDNN SM90 indexer requires seqlen_q <= seqlen_k * ratio, but the
        # compressor floors to seq_len // ratio blocks, so a seq_len that is not a
        # multiple of ratio leaves the last query token(s) without a compressed key
        # block. Right-pad to a multiple of ratio (the causal tail attends only real
        # tokens and is sliced off the output) so seqlen_k * ratio == seqlen_q.
        orig_seq_len = x.shape[1]
        ratio = self.compress_ratio
        pad = (-orig_seq_len) % ratio
        if pad:
            pos_tail = position_ids[:, -1:] + torch.arange(
                1, pad + 1, device=position_ids.device, dtype=position_ids.dtype
            )
            position_ids = torch.cat([position_ids, pos_tail], dim=1)
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            q = torch.nn.functional.pad(q, (0, 0, 0, pad))
            q_low = torch.nn.functional.pad(q_low, (0, 0, 0, pad))
            kv = torch.nn.functional.pad(kv, (0, 0, 0, pad))
        batch, seq_len, _ = x.shape
        compressed = self.compressor(
            x,
            position_ids=position_ids,
            rope_theta=self.config.compress_rope_theta,
        )
        index_comp = self.indexer.compressor(
            x,
            position_ids=position_ids,
            rope_theta=self.config.compress_rope_theta,
        )
        if compressed is None or index_comp is None:
            raise RuntimeError("DeepSeek V4 fused DSA requires at least one compressed KV entry.")
        compressed_kv = compressed.squeeze(1)
        index_k = index_comp.squeeze(1).transpose(0, 1).contiguous()
        kv_full = torch.cat([kv.squeeze(1), compressed_kv], dim=1)
        kv_full = kv_full.transpose(0, 1).contiguous()

        idx_cos, idx_sin = build_compressed_rope_cos_sin(
            position_ids,
            self.indexer.rope_head_dim,
            self.config.compress_rope_theta,
            config=self.config,
            use_yarn=self.compress_ratio > 1,
            device=x.device,
            dtype=x.dtype,
        )
        q_indexer = self.indexer.wq_b(q_low).view(
            batch, seq_len, self.indexer.index_n_heads, self.indexer.index_head_dim
        )
        q_indexer = q_indexer.transpose(1, 2)
        q_indexer = apply_partial_rope(q_indexer, idx_cos, idx_sin, self.indexer.rope_head_dim)
        q_indexer = rotate_activation(q_indexer)
        q_indexer = q_indexer.transpose(1, 2).transpose(0, 1).contiguous()
        weights_indexer = (
            (self.indexer.weights_proj(x).to(dtype=x.dtype) * (self.indexer.index_n_heads**-0.5))
            .transpose(0, 1)
            .contiguous()
        )
        indexer_topk = int(self.indexer.index_topk)
        if indexer_topk <= 0:
            raise RuntimeError("DeepSeek V4 fused DSA requires positive indexer_topk.")
        window_idxs = _window_topk_indices(
            batch,
            seq_len,
            self.config.sliding_window,
            device=x.device,
        )
        query = q.transpose(1, 2).transpose(0, 1).contiguous()
        sink = self.sinks.float()

        if self.training and torch.is_grad_enabled():
            out, _indexer_loss = dsa_kernels.fused_indexer_sparse_attn(
                query,
                kv_full,
                sink,
                window_idxs,
                q_indexer,
                index_k,
                weights_indexer,
                indexer_topk,
                self.compress_ratio,
                self.head_dim**-0.5,
                self.indexer.softmax_scale,
                0.0,
                sparse_loss=False,
                kv_offset=seq_len,
                calculate_per_token_loss=False,
            )
        else:
            topk_indices, _topk_length = dsa_kernels.indexer_topk(
                q_indexer,
                index_k,
                weights_indexer,
                indexer_topk,
                self.compress_ratio,
                indexer_softmax_scale=self.indexer.softmax_scale,
            )
            topk_indices = torch.where(
                topk_indices >= 0,
                topk_indices + seq_len,
                topk_indices,
            ).to(torch.int32)
            flat_idxs, flat_tlen = dsa_kernels.build_flat_topk_idxs(
                window_idxs,
                topk_indices,
                batch_size=batch,
                seqlen_kv=kv_full.size(0),
                compact=True,
            )
            out = dsa_kernels.dsa_sparse_attn(
                query,
                kv_full,
                sink,
                flat_idxs,
                self.head_dim**-0.5,
                topk_length=flat_tlen,
            )

        context = (
            out.view(seq_len, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()
        )
        if pad:
            context = context[:, :, :orig_seq_len, :].contiguous()
        return self._project_context(context, cos, sin)
