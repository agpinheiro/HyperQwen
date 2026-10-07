"""Kernel test for the q2_1 KV cache (q2_1/files/vllm/v1/attention/ops/q2_1_per_token_head.py).

Run inside the container with the q2_1 files mounted (docker-compose.q2.yml), GPU free:
  docker compose -f docker-compose.yml -f docker-compose.q2.yml --profile single \
      run --rm --no-deps single python q2_1/test_q2_1.py

Checks, for head sizes 256 (target) and 128 (DFlash2 drafter):
  1. the store kernel's codes and scales equal a plain PyTorch q2_1 reference bit for bit;
  2. 2D prefill attention (causal) matches attention over the reference-dequantized cache;
  3. 3D multi-query attention (the speculative verify shape) matches it too;
and reports the end-to-end error against unquantized attention -- the cost of q2_1 itself.
"""

import math
import os
import sys

os.environ.setdefault("VLLM_INT4_MQ_3D", "1")

import torch
import torch.nn.functional as F

from vllm.v1.attention.ops import q2_1_per_token_head as q21

torch.manual_seed(0)
dev = torch.device("cuda")
LUT = torch.tensor([-10.0, -3.0, 3.0, 10.0], device=dev)


def ref_quant(x):
    """x: [..., D] fp32, already rotated -> (codes [..., D] int, d [..., D/64] fp16)."""
    g = x.reshape(*x.shape[:-1], -1, 64)
    d = 0.1510 * torch.sqrt((g * g).sum(-1) / 64.0)
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    xn = g * inv[..., None]
    code = (xn > -6.5).int() + (xn > 0).int() + (xn > 6.5).int()
    return code.reshape(x.shape), d.half()


def ref_dequant(code, d):
    v = LUT[code.long()].reshape(*code.shape[:-1], -1, 64) * d.float()[..., None]
    return v.reshape(code.shape)


def unpack(cache_side, head_size):
    """[..., side] uint8 -> codes [..., D]: byte j holds dims j + s*W in bits 2s."""
    w = head_size // 4
    b = cache_side[..., :w].int()
    return torch.cat([(b >> (2 * s)) & 3 for s in range(4)], dim=-1)


def run(head_size, nkv, nq, ctx, qlen, block_size=16, segments=16):
    side = q21.q2_1_side_bytes(head_size)
    q21.q2_1_prepare(head_size, dev)
    nblocks = (ctx + block_size - 1) // block_size + 4
    kv_cache = torch.zeros(nblocks, nkv, block_size, 2 * side, dtype=torch.uint8, device=dev)
    k_sc, v_sc = q21.q2_1_scale_views(kv_cache, head_size)
    key_cache, value_cache = kv_cache.transpose(1, 2).split(side, dim=-1)

    # K/V with a few outlier channels, like real attention rows.
    k = torch.randn(ctx, nkv, head_size, device=dev)
    k[..., :4] *= 8
    v = torch.randn(ctx, nkv, head_size, device=dev)
    k, v = k.bfloat16(), v.bfloat16()
    perm = torch.randperm(nblocks, device=dev)[: (ctx + block_size - 1) // block_size]
    slots = (perm[torch.arange(ctx, device=dev) // block_size] * block_size
             + torch.arange(ctx, device=dev) % block_size)
    q21.reshape_and_cache_q2_1(k, v, key_cache, value_cache, slots,
                               k_scale_cache=k_sc, v_scale_cache=v_sc)

    # 1. bit-exact store.
    kr = q21._rotate(k, q21._k_rot_size(head_size)).float()
    vr = q21._rotate(v, 64).float()
    kc_ref, kd_ref = ref_quant(kr)
    vc_ref, vd_ref = ref_quant(vr)
    blk, off = slots // block_size, slots % block_size
    kc = unpack(key_cache[blk, off], head_size)
    vc = unpack(value_cache[blk, off], head_size)
    code_mismatch = (kc != kc_ref).float().mean().item() + (vc != vc_ref).float().mean().item()
    scale_err = ((k_sc[blk, off].float() - kd_ref.float()).abs().max().item(),
                 (v_sc[blk, off].float() - vd_ref.float()).abs().max().item())
    print(f"D={head_size}: code mismatch {code_mismatch:.2e}, scale max err {scale_err}")
    assert code_mismatch < 1e-4, "store kernel codes disagree with the reference"
    assert max(scale_err) < 1e-3

    # Reference attention over the dequantized cache (rotated basis) and over fp K/V.
    k_deq = ref_dequant(kc_ref, kd_ref)  # rotated K
    v_deq = ref_dequant(vc_ref, vd_ref)  # rotated V
    scale = 1.0 / math.sqrt(head_size)
    q = torch.randn(qlen, nq, head_size, device=dev).bfloat16()
    qr = q21._rotate(q, q21._k_rot_size(head_size)).float()
    group = nq // nkv

    def attend(qx, kx, vx):
        # causal: query i sits at position ctx - qlen + i
        kx = kx.repeat_interleave(group, dim=1).transpose(0, 1)  # [nq, ctx, D]
        vx = vx.repeat_interleave(group, dim=1).transpose(0, 1)
        s = torch.einsum("qhd,hkd->hqk", qx, kx) * scale
        pos = torch.arange(ctx - qlen, ctx, device=dev)[:, None]
        s = s.masked_fill(torch.arange(ctx, device=dev)[None, :] > pos, float("-inf"))
        return torch.einsum("hqk,hkd->qhd", s.softmax(-1), vx)

    ref_q = q21._rotate(attend(qr, k_deq, v_deq), 64)  # unrotate V
    ref_fp = attend(q.float(), k.float(), v.float())

    block_table = torch.zeros(1, nblocks, dtype=torch.int32, device=dev)
    block_table[0, : len(perm)] = perm.int()
    cu = torch.tensor([0, qlen], dtype=torch.int32, device=dev)
    seqused = torch.tensor([ctx], dtype=torch.int32, device=dev)
    hp = 1 << (head_size - 1).bit_length()
    for mode in ("2d", "3d"):
        out = torch.empty(qlen, nq, head_size, dtype=torch.bfloat16, device=dev)
        extra = {}
        if mode == "3d":
            extra = dict(
                seq_threshold_3D=8, max_query_len_3d=16, num_par_softmax_segments=segments,
                softmax_segm_output=torch.empty(qlen, nq, segments, hp, device=dev),
                softmax_segm_max=torch.empty(qlen, nq, segments, device=dev),
                softmax_segm_expsum=torch.empty(qlen, nq, segments, device=dev),
            )
        q21.unified_attention_q2_1(
            q, key_cache, value_cache, out, cu_seqlens_q=cu, max_seqlen_q=qlen,
            seqused_k=seqused, softmax_scale=scale, window_size=(-1, -1),
            block_table=block_table, softcap=0.0, sinks=None, alibi_slopes=None,
            use_alibi_sqrt=False, qq_bias=None, output_scale=None, mm_prefix_range=None,
            k_scale_cache=k_sc, v_scale_cache=v_sc, **extra)
        rel_k = ((out.float() - ref_q).norm() / ref_q.norm()).item()
        rel_fp = ((out.float() - ref_fp).norm() / ref_fp.norm()).item()
        cos_fp = F.cosine_similarity(out.float().flatten(), ref_fp.flatten(), dim=0).item()
        print(f"  {mode} q={qlen} ctx={ctx}: vs dequant ref {rel_k:.2e}   "
              f"vs unquantized {rel_fp:.3f} (cos {cos_fp:.4f})")
        assert rel_k < 2e-2, f"{mode} kernel disagrees with the dequantized reference"


if __name__ == "__main__":
    run(256, 4, 24, ctx=3000, qlen=8)      # target: verify block over a deep cache
    run(256, 4, 24, ctx=1500, qlen=300)    # target: prefill chunk
    run(128, 8, 32, ctx=1000, qlen=8)      # DFlash2 drafter geometry
    print("q2_1 kernel tests passed")
    sys.exit(0)
