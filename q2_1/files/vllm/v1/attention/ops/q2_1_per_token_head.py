# SPDX-License-Identifier: Apache-2.0
"""llama.cpp's q2_1 KV cache format as a per-token-head mode of TRITON_ATTN.

Format, per token, per KV head, per side (K and V), head size D (a multiple of 64):

  - D/4 bytes of 2-bit codes, then D/64 fp16 scales: 2.25 bits per value, 576 bytes per
    token per layer for this model's 4 KV heads x 256 (K+V), against 840 for KVarN k4v2
    and 1056 for int4_per_token_head.
  - A code c in 0..3 decodes to {-10, -3, +3, +10}[c] * d, with d = 0.1510 * rms of its
    64-value group: the Lloyd-Max levels for Gaussian data (+/-0.4528, +/-1.5104 sigma).
    Encoding is against the thresholds -6.5, 0, +6.5 in units of d. Same rule, same
    constants as quantize_f32_q2_1_block in the turboquant llama.cpp fork.
  - Byte j holds the codes of dims j, j+W, j+2W, j+3W (W = D/4) in bits 0-1, 2-3, 4-5,
    6-7. One byte load therefore yields four contiguous W-wide streams, and each stream
    lies inside a single scale group (W = 64 for D = 256, 32 for D = 128), so the scales
    fold in after each stream's dot product exactly -- the int8 per-token-head trick --
    and the hot loop never multiplies a scale per element.
  - Before quantization K is rotated by an orthonormal Walsh-Hadamard of the largest
    power-of-two size dividing D, and Q gets the same rotation, so q.k is unchanged;
    V is rotated by a 64-wide block Hadamard, undone on the attention output (H.H = I).
    That is llama.cpp's attn_rot_k / attn_rot_v, and it is what makes Gaussian
    Lloyd-Max levels a good fit for real K/V rows with outlier channels.

The read kernel is int4_per_token_head's _attn_packed with four streams in place of two:
the same 2D prefill / 3D split-KV dispatch (multi-query 3D for the speculative verify
behind VLLM_INT4_MQ_3D, which this mode shares), the same causal / per-seq-causal /
sliding-window handling the DFlash2 drafter's layers need.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments

# d = Q2_1_SCALE * rms(group); codes {-10,-3,3,10}; thresholds -6.5 / 0 / 6.5.
Q2_1_SCALE = 0.1510
Q2_1_GROUP = 64


def q2_1_side_bytes(head_size: int) -> int:
    """Bytes for one side (K or V) of one (token, head): codes + fp16 scales."""
    assert head_size % Q2_1_GROUP == 0, f"q2_1 needs head_size % 64 == 0, got {head_size}"
    return head_size // 4 + 2 * (head_size // Q2_1_GROUP)


# ----------------------------------------------------------------------
# Rotations
# ----------------------------------------------------------------------

_HADAMARD: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _k_rot_size(head_size: int) -> int:
    """llama.cpp build_input_k_rot: the largest power of two dividing head_size."""
    return head_size & -head_size


def _hadamard(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, device, dtype)
    h = _HADAMARD.get(key)
    if h is None:
        m = torch.ones(1, 1, dtype=torch.float64)
        while m.shape[0] < n:
            m = torch.cat([torch.cat([m, m], 1), torch.cat([m, -m], 1)], 0)
        h = (m / math.sqrt(n)).to(device=device, dtype=dtype)
        _HADAMARD[key] = h
    return h


def q2_1_prepare(head_size: int, device: torch.device) -> None:
    """Build the rotation matrices ahead of time (never inside CUDA graph capture)."""
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        _hadamard(_k_rot_size(head_size), device, dtype)
        _hadamard(Q2_1_GROUP, device, dtype)


def _rotate(x: torch.Tensor, n: int) -> torch.Tensor:
    """x @ blockdiag(H_n) over the last dim, H_n orthonormal and symmetric."""
    h = _hadamard(n, x.device, x.dtype)
    return torch.matmul(x.reshape(-1, n), h).reshape(x.shape)


# ----------------------------------------------------------------------
# Cache layout: scale views over the bytes after each side's codes
# ----------------------------------------------------------------------


def q2_1_scale_views(
    kv_cache: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp16 views of the K and V scales, shape (num_blocks, block_size, nkv, D/64).

    ``kv_cache`` is the packed logical uint8 tensor (num_blocks, nkv, block_size,
    2 * side) with HND or NHD physical strides; each side is [codes D/4 | scales].
    """
    num_blocks, nkv, block_size, content = kv_cache.shape
    assert kv_cache.element_size() == 1
    side = content // 2
    packed = head_size // 4
    ng = head_size // Q2_1_GROUP
    assert side == q2_1_side_bytes(head_size), (side, head_size)

    base = torch.tensor([], dtype=torch.float16, device=kv_cache.device).set_(
        kv_cache.untyped_storage()
    )

    def halves(nbytes: int) -> int:
        assert nbytes % 2 == 0, "q2_1 scale views need 2-byte aligned strides"
        return nbytes // 2

    st = kv_cache.stride()
    off = kv_cache.storage_offset()

    def view(byte_offset: int) -> torch.Tensor:
        return torch.as_strided(
            base,
            size=(num_blocks, block_size, nkv, ng),
            stride=(halves(st[0]), halves(st[2]), halves(st[1]), 1),
            storage_offset=halves(off + byte_offset),
        )

    k_scales = view(packed)
    v_scales = view(side + packed)
    k_scales.zero_()
    v_scales.zero_()
    return k_scales, v_scales


# ----------------------------------------------------------------------
# Write path
# ----------------------------------------------------------------------


@triton.jit
def _q2_1_quant_store(
    src_ptr,
    cache_ptr,
    scale_ptr,
    tok,
    head,
    blk,
    slot_in_blk,
    stride_src_tok,
    stride_src_head,
    stride_c_blk,
    stride_c_slot,
    stride_c_head,
    stride_s_blk,
    stride_s_slot,
    stride_s_head,
    W: tl.constexpr,
    SPG: tl.constexpr,
):
    s = tl.arange(0, 4)
    j = tl.arange(0, W)
    x = tl.load(
        src_ptr + tok * stride_src_tok + head * stride_src_head + s[:, None] * W + j[None, :]
    ).to(tl.float32)  # [4, W]: stream s = dims s*W .. s*W+W-1
    # Sum of squares per 64-value group, broadcast back to its SPG streams.
    ss = tl.sum(x * x, axis=1)
    same_group = (s[:, None] // SPG) == (s[None, :] // SPG)
    ss_group = tl.sum(tl.where(same_group, ss[None, :], 0.0), axis=1)
    d = 0.1510 * tl.sqrt(ss_group / 64.0)
    inv_d = tl.where(d > 0.0, 1.0 / d, 0.0)
    xn = x * inv_d[:, None]
    code = (xn > -6.5).to(tl.int32) + (xn > 0.0).to(tl.int32) + (xn > 6.5).to(tl.int32)
    packed = tl.sum(code << (2 * s)[:, None], axis=0).to(tl.uint8)
    tl.store(
        cache_ptr + blk * stride_c_blk + slot_in_blk * stride_c_slot + head * stride_c_head + j,
        packed,
    )
    tl.store(
        scale_ptr
        + blk * stride_s_blk
        + slot_in_blk * stride_s_slot
        + head * stride_s_head
        + s // SPG,
        d.to(tl.float16),
        mask=(s % SPG) == 0,
    )


@triton.jit
def _reshape_cache_q2_1_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    slot_mapping_ptr,
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_kc_blk: tl.int64,
    stride_kc_slot: tl.int64,
    stride_kc_head: tl.int64,
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    block_size: tl.constexpr,
    W: tl.constexpr,
    SPG: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return
    blk = slot // block_size
    slot_in_blk = slot % block_size
    _q2_1_quant_store(
        key_ptr, key_cache_ptr, k_scale_ptr, tok, head, blk, slot_in_blk,
        stride_key_tok, stride_key_head,
        stride_kc_blk, stride_kc_slot, stride_kc_head,
        stride_ks_blk, stride_ks_slot, stride_ks_head,
        W, SPG,
    )
    _q2_1_quant_store(
        value_ptr, value_cache_ptr, v_scale_ptr, tok, head, blk, slot_in_blk,
        stride_val_tok, stride_val_head,
        stride_vc_blk, stride_vc_slot, stride_vc_head,
        stride_vs_blk, stride_vs_slot, stride_vs_head,
        W, SPG,
    )


def reshape_and_cache_q2_1(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> None:
    """Rotate, quantize to q2_1 and write into the paged cache."""
    num_tokens, num_kv_heads, head_size = key.shape
    assert value.shape[2] == head_size and head_size % Q2_1_GROUP == 0
    key = _rotate(key, _k_rot_size(head_size))
    value = _rotate(value, Q2_1_GROUP)
    w = head_size // 4
    _reshape_cache_q2_1_kernel[(num_tokens, num_kv_heads)](
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        k_scale_cache.stride(0),
        k_scale_cache.stride(1),
        k_scale_cache.stride(2),
        v_scale_cache.stride(0),
        v_scale_cache.stride(1),
        v_scale_cache.stride(2),
        block_size=key_cache.shape[1],
        W=w,
        SPG=Q2_1_GROUP // w,
        num_warps=4,
    )


# ----------------------------------------------------------------------
# Read path
# ----------------------------------------------------------------------


@triton.jit
def _q2_1_decode(c):
    """2-bit code -> codebook level {-10, -3, +3, +10} (exact in bf16)."""
    return tl.where(c >= 2, tl.where(c == 3, 10.0, 3.0), tl.where(c == 0, -10.0, -3.0))


@triton.jit
def _attn_q2_1(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    scale,
    softcap,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    W: tl.constexpr,  # HEAD_SIZE // 4: stream width = packed bytes per side
    SPG: tl.constexpr,  # streams per 64-value scale group (1 for D=256, 2 for D=128)
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    IS_3D: tl.constexpr,
    USE_CAUSAL: tl.constexpr = True,
    USE_PER_SEQ_CAUSAL: tl.constexpr = False,
    per_seq_causal_ptr=None,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, TILE_SIZE)
    offs_w = tl.arange(0, W)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Four W-wide streams of (rotated) Q, one per 2-bit field of the packed byte.
    q_base = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_w[None, :]
    )
    q_mask = query_mask_0[:, None] & query_mask_1[:, None]
    Q0 = tl.load(query_ptr + q_base, mask=q_mask, other=0.0).to(tl.bfloat16)
    Q1 = tl.load(query_ptr + q_base + W, mask=q_mask, other=0.0).to(tl.bfloat16)
    Q2 = tl.load(query_ptr + q_base + 2 * W, mask=q_mask, other=0.0).to(tl.bfloat16)
    Q3 = tl.load(query_ptr + q_base + 3 * W, mask=q_mask, other=0.0).to(tl.bfloat16)

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_M, W], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, W], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, W], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_M, W], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
    )

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)
        slot_in_blk = seq_offset % BLOCK_SIZE

        # One packed byte per (dim j, token): all four streams in a single load.
        k_off = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_w[:, None] * stride_k_cache_3
            + slot_in_blk[None, :] * stride_k_cache_1
        )
        K_packed = tl.load(key_cache_ptr + k_off, mask=tile_mask[None, :], other=0)
        v_off = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_w[None, :] * stride_v_cache_3
            + slot_in_blk[:, None] * stride_v_cache_1
        )
        V_packed = tl.load(value_cache_ptr + v_off, mask=tile_mask[:, None], other=0)

        ks_base = (
            k_scale_cache_ptr
            + physical_block_idx * stride_ks_blk
            + slot_in_blk * stride_ks_slot
            + kv_head_idx * stride_ks_head
        )
        vs_base = (
            v_scale_cache_ptr
            + physical_block_idx * stride_vs_blk
            + slot_in_blk * stride_vs_slot
            + kv_head_idx * stride_vs_head
        )
        ks0 = tl.load(ks_base + 0 // SPG, mask=tile_mask, other=0.0).to(tl.float32)
        ks1 = tl.load(ks_base + 1 // SPG, mask=tile_mask, other=0.0).to(tl.float32)
        ks2 = tl.load(ks_base + 2 // SPG, mask=tile_mask, other=0.0).to(tl.float32)
        ks3 = tl.load(ks_base + 3 // SPG, mask=tile_mask, other=0.0).to(tl.float32)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            USE_CAUSAL,
            USE_PER_SEQ_CAUSAL,
            per_seq_causal_ptr,
        )

        # Score: per-stream dot on the raw codebook levels, the stream's (token) scale
        # folded in after -- exact, since it is constant across the stream.
        K0 = _q2_1_decode(K_packed & 3).to(tl.bfloat16)
        S = tl.dot(Q0, K0) * ks0[None, :]
        K1 = _q2_1_decode((K_packed >> 2) & 3).to(tl.bfloat16)
        S += tl.dot(Q1, K1) * ks1[None, :]
        K2 = _q2_1_decode((K_packed >> 4) & 3).to(tl.bfloat16)
        S += tl.dot(Q2, K2) * ks2[None, :]
        K3 = _q2_1_decode((K_packed >> 6) & 3).to(tl.bfloat16)
        S += tl.dot(Q3, K3) * ks3[None, :]
        S = S * scale

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            sw_mask = (context_len + qpos_lo - seq_offset) < SLIDING_WINDOW
            P = tl.where(sw_mask[None, :], P, 0.0)

        vs0 = tl.load(vs_base + 0 // SPG, mask=tile_mask, other=0.0).to(tl.float32)
        vs1 = tl.load(vs_base + 1 // SPG, mask=tile_mask, other=0.0).to(tl.float32)
        vs2 = tl.load(vs_base + 2 // SPG, mask=tile_mask, other=0.0).to(tl.float32)
        vs3 = tl.load(vs_base + 3 // SPG, mask=tile_mask, other=0.0).to(tl.float32)

        V0 = _q2_1_decode(V_packed & 3).to(tl.bfloat16)
        acc0 = acc0 * alpha[:, None] + tl.dot((P * vs0[None, :]).to(tl.bfloat16), V0)
        V1 = _q2_1_decode((V_packed >> 2) & 3).to(tl.bfloat16)
        acc1 = acc1 * alpha[:, None] + tl.dot((P * vs1[None, :]).to(tl.bfloat16), V1)
        V2 = _q2_1_decode((V_packed >> 4) & 3).to(tl.bfloat16)
        acc2 = acc2 * alpha[:, None] + tl.dot((P * vs2[None, :]).to(tl.bfloat16), V2)
        V3 = _q2_1_decode((V_packed >> 6) & 3).to(tl.bfloat16)
        acc3 = acc3 * alpha[:, None] + tl.dot((P * vs3[None, :]).to(tl.bfloat16), V3)

    out_mask = query_mask_0[:, None] & query_mask_1[:, None]
    if IS_3D:
        segm_base = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
            + offs_w[None, :]
        )
        tl.store(segm_output_ptr + segm_base, acc0, mask=out_mask)
        tl.store(segm_output_ptr + segm_base + W, acc1, mask=out_mask)
        tl.store(segm_output_ptr + segm_base + 2 * W, acc2, mask=out_mask)
        tl.store(segm_output_ptr + segm_base + 3 * W, acc3, mask=out_mask)
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        out_base = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_w[None, :]
        )
        tl.store(output_ptr + out_base, acc0 / L[:, None], mask=out_mask)
        tl.store(output_ptr + out_base + W, acc1 / L[:, None], mask=out_mask)
        tl.store(output_ptr + out_base + 2 * W, acc2 / L[:, None], mask=out_mask)
        tl.store(output_ptr + out_base + 3 * W, acc3 / L[:, None], mask=out_mask)


def _use_3d(
    q, max_seqlen_q, num_seqs, seq_threshold_3D, max_query_len_3d,
    num_par_softmax_segments, softmax_segm_output, softmax_segm_max, softmax_segm_expsum,
) -> bool:
    """int4_per_token_head's split-KV eligibility, multi-query 3D included (the verify
    step of speculative decoding is a multi-query batch). Gated by the same
    VLLM_INT4_MQ_3D switch; the 2D path is the correct fallback either way."""
    scratch = (
        softmax_segm_output is not None
        and softmax_segm_max is not None
        and softmax_segm_expsum is not None
    )
    if seq_threshold_3D is None or num_par_softmax_segments is None or not scratch:
        return False
    if num_seqs > seq_threshold_3D or envs.VLLM_BATCH_INVARIANT:
        return False
    if max_seqlen_q > 1:
        qlen_cap = max_query_len_3d if max_query_len_3d is not None else 16
        capacity_ok = (
            q.shape[0] <= softmax_segm_output.shape[0]
            and q.shape[0] <= softmax_segm_max.shape[0]
            and q.shape[0] <= softmax_segm_expsum.shape[0]
        )
        return bool(envs.VLLM_INT4_MQ_3D) and max_seqlen_q <= qlen_cap and capacity_ok
    return True


def unified_attention_q2_1(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    window_size: tuple[int, int],
    block_table: torch.Tensor,
    softcap: float,
    sinks: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    use_alibi_sqrt: bool,
    qq_bias: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    mm_prefix_range: torch.Tensor | None,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    seq_threshold_3D: int | None = None,
    max_query_len_3d: int | None = None,
    scratch_token_capacity_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    use_causal: bool = True,
    per_seq_causal_ptr: torch.Tensor | None = None,
) -> None:
    """Paged attention over the q2_1 cache, writing into *out*."""
    from vllm.v1.attention.ops.triton_unified_attention import _get_tile_size

    # The output is produced in the V-rotated basis and rotated back here; an fp8
    # output would have to be dequantized first, which nothing on this path needs.
    assert output_scale is None, "q2_1 KV cache does not fuse fp8 output quantization"
    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    head_size = q.shape[2]
    q = _rotate(q, _k_rot_size(head_size))

    use_mm_prefix = mm_prefix_range is not None
    max_mm_ranges = mm_prefix_range.shape[1] if use_mm_prefix else 0

    block_size = v_cache.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    use_3d = _use_3d(
        q, max_seqlen_q, num_seqs, seq_threshold_3D, max_query_len_3d,
        num_par_softmax_segments, softmax_segm_output, softmax_segm_max,
        softmax_segm_expsum,
    )
    tile_size = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=not use_3d
    )
    if use_3d:
        grid: tuple[Any, ...] = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
    else:
        grid = (total_num_q_blocks, num_kv_heads)

    w = head_size // 4
    _attn_q2_1[grid](
        output_ptr=out,
        segm_output_ptr=softmax_segm_output if use_3d else out,
        segm_max_ptr=softmax_segm_max if use_3d else out,
        segm_expsum_ptr=softmax_segm_expsum if use_3d else out,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        scale=softmax_scale,
        softcap=softcap,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if qq_bias is not None else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        W=w,
        SPG=Q2_1_GROUP // w,
        USE_ALIBI_SLOPES=alibi_slopes is not None,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=qq_bias is not None,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=(1 + window_size[0]),
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments if use_3d else 1,
        IS_3D=use_3d,
        USE_CAUSAL=use_causal,
        USE_PER_SEQ_CAUSAL=(per_seq_causal_ptr is not None),
        per_seq_causal_ptr=per_seq_causal_ptr,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=tile_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=False,
        )

    # Undo the V rotation (64-wide blocks, H.H = I).
    out.copy_(_rotate(out, Q2_1_GROUP))
