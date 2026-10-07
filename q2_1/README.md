# q2_1 KV cache for vLLM (262K context on one RTX 3090)

llama.cpp's `q2_1` KV cache type, ported to vLLM's TRITON_ATTN backend as a new
`--kv-cache-dtype q2_1`. At 2.25 bits per value, one 24 GB card holds a 549,694-token
pool next to the W4A16 27B model and the DFlash2 drafter: two full 262,144-token
requests at once.

## Format

The math is llama.cpp's `quantize_row_q2_1_ref` (ggml-quants.c) unchanged:

- groups of 64 values, one fp16 scale per group, `d = 0.1510 * rms`;
- codes `(x/d > -6.5) + (x/d > 0) + (x/d > 6.5)`, levels `{-10, -3, +3, +10} * d`
  (the Lloyd-Max optimum for Gaussian data);
- the same rotations llama.cpp applies to the KV cache: a Walsh-Hadamard transform
  over the largest power of two dividing `head_dim` on K (and Q), and a 64-wide block
  Hadamard on V, undone on the attention output.

Only the byte layout differs. Per token, head and side (K or V), a `head_dim`-wide
row takes `head_dim/4` bytes of codes followed by `head_dim/64` fp16 scales (72 bytes
for head 256). Byte `j` holds dims `j`, `j+W`, `j+2W`, `j+3W` (`W = head_dim/4`) in
bits `2s`, so the attention kernel unpacks each byte into four streams that each
sit inside a single scale group. Each stream is a plain tensor-core `tl.dot`, and the
scale is applied once per dot instead of per element.

## Files

| Path | What |
|---|---|
| `files/vllm/v1/attention/ops/q2_1_per_token_head.py` | the new module: rotations, store kernel, 2D/3D attention kernel |
| `files/vllm/...` (6 others) | plumbing: dtype name, `KVQuantMode.Q2_1`, page size, dispatch |
| `vllm-q2_1.diff` | the plumbing changes as a diff against vLLM 0.30 in the image |
| `alternative.sh` | the image's `single-user/alternative.sh` with a `KV_DTYPE` switch |
| `test_q2_1.py` | kernel test |

The files are bind-mounted over the image's copies by
[`docker-compose.override.yml`](../docker-compose.override.yml). Docker Compose merges
that file automatically, so q2_1 is the default for `--profile single`; nothing is rebuilt.

## Run

```bash
docker compose --profile single up -d
```

The defaults are `KV_DTYPE=q2_1`, `MAX_LEN=262144` and `GPU_UTIL=0.93`. The pool is
sized from `GPU_UTIL`, not from `KV_MEM`/`CTX`. Any of these can be set in `.env`.
To compare against the stock int4 cache, set `KV_DTYPE=int4_per_token_head`; it runs
the same launcher. `docker compose -f docker-compose.yml --profile single up -d`
skips the overlay entirely.

Kernel test (GPU free, server stopped; `MSYS_NO_PATHCONV=1` keeps Git Bash from
rewriting the `/app` paths):

```bash
MSYS_NO_PATHCONV=1 docker compose --profile single run --rm --no-deps single \
    /app/venv/bin/python /app/q2_1/test_q2_1.py
```

It checks that the store kernel's codes equal a PyTorch q2_1 reference bit for bit,
with scales within one fp16 ulp (llama.cpp sums squares with sequential `fmaf`, the
kernel with a tree). It also checks that 2D prefill and 3D speculative-verify
attention land within ~3e-3 of exact attention over the dequantized cache, for
head 256 (target) and head 128 (DFlash2 drafter).

## Measured

One RTX 3090 that also drives the display, WSL2, `SPEC=dflash2`, temp 0, thinking
off, 512-token answers, `MAX_LEN=262144`:

| | code | prose |
|---|---:|---:|
| **q2_1** (pool 549,694) | 197.7 | 111.4 |
| `CTX=huge` KVarN, KV_MEM trimmed (pool 263,663) | 203.8 | 106.9 |
| `CTX=fast` int4, 65K context (reference) | 226.8 | 109.5 |

Decode tok/s on a source-code prompt padded to each length (second of two runs):

| context | q2_1 | KVarN | q2_1 cold prefill | KVarN cold prefill |
|---:|---:|---:|---:|---:|
| 7K | 174.5 | 125.1 | 7.1 s | |
| 55K | 81.2 | 45.1 | 85.7 s | 63.9 s |
| 113K | 57.7 | 27.5 | 267 s | 157 s |
| 237K | 34.9 | crashed | 945 s | |

Shared GPU memory stayed under 160 MiB throughout, so nothing paged to system RAM.
Cold prefill (a whole prompt at once into an empty cache) is q2_1's weak spot: the
2D prefill kernel still has headroom. In a normal conversation, prefix caching keeps
the history, so each turn prefills only its new input over the cached context.

## Quality

The quantizer is llama.cpp's, so llama.cpp's perplexity measurement carries over
(ctx 8192, 4 chunks of War and Peace, Qwen3.8-27B-Q4_K_M):

| KV | bits/value | PPL | vs f16 |
|---|---:|---:|---:|
| f16 | 16 | 6.8594 | |
| q4_0 | 4.5 | 6.8701 | +0.16 % |
| **q2_1** | **2.25** | **7.1927** | **+4.86 %** |

In llama.cpp the same q2_1 cache also recalled a fact planted ~428K tokens back
(with YaRN). Perplexity has not been re-measured on the vLLM side.

Draft acceptance in vLLM, from the `/metrics` spec-decode counters. Each prompt gets
a 512-token greedy answer and the drafter proposes 7 tokens per step. Acceptance
length counts tokens per step, including the target's own token:

| prompt | accept length | rate |
|---|---:|---:|
| code (4 prompts: Python, Go, TypeScript, SQL) | 3.5–5.2 | 35–61 % |
| prose (3 prompts, one in Portuguese) | 2.6–3.1 | 22–30 % |
| code question over 7K / 55K / 105K of C source | 5.2 / 4.5 / 4.7 | 61 / 50 / 52 % |

Acceptance holds up with depth: past 55K it does not keep falling. No KVarN or int4
acceptance was measured on the same prompts, so this table is not an A/B.
