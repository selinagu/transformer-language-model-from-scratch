# Benchmark protocol

The benchmark suite measures the current implementation before optimization. Do not compare
results across different devices, dtypes, model shapes, or commits without recording that
metadata.

Every CSV row records the Git commit, PyTorch/Python versions, device, dtype, warmup count,
measurement count, median latency, mean latency, throughput, and benchmark-specific details.

## Attention baseline

Run the hand-written attention implementation:

```bash
uv run python -m benchmarks.attention \
  --contexts 128 256 512 1024 \
  --device cuda \
  --output benchmarks/results/attention.csv
```

This is the reference baseline for the later SDPA backend.

## Generation baseline

Run full-prefix decoding, which intentionally recomputes the complete sequence for every
generated token:

```bash
uv run python -m benchmarks.generation \
  --prompt-length 32 \
  --max-new-tokens 128 \
  --device cuda \
  --output benchmarks/results/generation.csv
```

The later KV-cache benchmark must use the same model dimensions, prompt length, generated
length, device, dtype, warmup count, and iteration count.

## Tokenizer baseline

```bash
uv run python -m benchmarks.tokenizer \
  --vocab-path outputs/tinystories_vocab_10000.pkl \
  --merges-path outputs/tinystories_merges_10000.pkl \
  --input-path data/TinyStoriesV2-GPT4-valid.txt \
  --output benchmarks/results/tokenizer.csv
```

## Measurement rules

- Treat the defaults as smoke-test settings; use at least 50 measurement iterations for formal
  GPU results.
- Do not commit a performance claim without its CSV result and hardware details.
- CUDA timing is synchronized before and after each iteration so asynchronous kernel execution
  is included in the latency measurement.
- Benchmark scripts measure inference only and run under `torch.inference_mode()`.
