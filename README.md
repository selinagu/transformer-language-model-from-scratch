# Mini GPT from Scratch

## Overview

A compact, from-scratch GPT-style language-model pipeline in PyTorch:

~~~text
raw UTF-8 text → byte-level BPE → tokenized NumPy data
→ decoder-only Transformer → checkpoint → HTTP generation
~~~

The repository includes a verified CPU smoke path that creates all artifacts locally. It does not include a pretrained checkpoint or a dataset.

## Engineering Highlights

- Byte-level BPE training, artifact export, and deterministic tokenizer fingerprints.
- Config-driven Transformer training with RMSNorm, RoPE, SwiGLU, causal attention, AdamW, warmup/cosine learning rate scheduling, and gradient clipping.
- Checkpoints carry model, optimizer, RNG, model configuration, and tokenizer-identity metadata.
- Resume restores Python, NumPy, PyTorch CPU, and available CUDA RNG state.
- FastAPI startup rejects missing, legacy, vocabulary-mismatched, special-token-mismatched, or fingerprint-mismatched artifacts.
- SDPA and KV-cache inference paths are covered by regression tests.

## Architecture

~~~text
train_bpe.py       raw text → vocab.pkl + merges.pkl
prepare_data.py    raw text + tokenizer → train.npy + validation.npy + metadata JSON
train.py           config + NPY data + tokenizer → checkpoint.pt
service.py         compatible checkpoint + tokenizer → /health and /generate
~~~

## Results

The included tiny CPU workflow is a correctness smoke test, not a quality benchmark. It trains only two steps and makes no claim about generation quality.

A TinyStories run was conducted historically, but its quantitative outputs were not re-verified as part of this reproducibility workflow; no historical metrics or sample generations are asserted here.

## Quick Start

Install the locked project environment:

~~~bash
uv sync
mkdir -p artifacts
printf "a tiny story begins <|endoftext|> a tiny story continues\n" > artifacts/train.txt
printf "a tiny validation story <|endoftext|> continues\n" > artifacts/validation.txt
~~~

Create BPE artifacts. The vocabulary size must match the training config:

~~~bash
uv run python -m cs336_basics.train_bpe   --input-path artifacts/train.txt   --vocab-size 264   --vocab-path artifacts/vocab.pkl   --merges-path artifacts/merges.pkl   --special-token "<|endoftext|>"   --num-processes 1
~~~

Tokenize both raw text files with those exact artifacts:

~~~bash
uv run python -m cs336_basics.prepare_data   --train-text-path artifacts/train.txt   --validation-text-path artifacts/validation.txt   --vocab-path artifacts/vocab.pkl   --merges-path artifacts/merges.pkl   --train-output-path artifacts/train.npy   --validation-output-path artifacts/validation.npy   --special-token "<|endoftext|>"
~~~

The command writes two uint32 NPY files and artifacts/train.npy.metadata.json, including token counts, dataset hashes, and the tokenizer fingerprint.

Train the tiny CPU model:

~~~bash
uv run python -m cs336_basics.train   --config configs/tiny_cpu_smoke.yaml   --train_data artifacts/train.npy   --valid_data artifacts/validation.npy   --vocab_path artifacts/vocab.pkl   --merges_path artifacts/merges.pkl   --checkpoint_path artifacts/checkpoint.pt   --log_path artifacts/train.csv   --device cpu
~~~

For larger experiments, choose a configuration whose model.vocab_size exactly equals the BPE vocabulary size and retain the checkpoint, vocabulary, and merges together.

The CLI entrypoints above are the generic, reproducible workflow. The TinyStories and OpenWebText helper scripts are retained as historical experiment helpers with repository-relative data and output paths; they are not the recommended setup path.

## HTTP Inference & Docker

Start the verified artifacts locally:

~~~bash
uv run python -m cs336_basics.service   --checkpoint-path artifacts/checkpoint.pt   --vocab-path artifacts/vocab.pkl   --merges-path artifacts/merges.pkl   --device cpu
~~~

In another terminal:

~~~bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate   -H "content-type: application/json"   -d "{\"prompt\":\"a tiny story\",\"max_new_tokens\":8,\"temperature\":0}"
~~~

The Docker image installs the CPU service runtime requirements and uses CPU-only PyTorch. Build it, then replace <repo-root> with this checkout absolute path:

~~~bash
docker build -t mini-gpt .
docker run --rm -p 8000:8000   -v "<repo-root>/artifacts/checkpoint.pt:/models/checkpoint.pt:ro"   -v "<repo-root>/artifacts/vocab.pkl:/models/vocab.pkl:ro"   -v "<repo-root>/artifacts/merges.pkl:/models/merges.pkl:ro"   mini-gpt
~~~

The container validates mounted artifacts at startup. Missing files and incompatible tokenizer artifacts fail with a clear startup error.

Docker execution has not yet been verified in the current WSL environment because Docker Desktop integration is unavailable there.

## Testing and Reliability

~~~bash
uv run pytest -q
uv run ruff check .
uv lock --check
~~~

Tests cover BPE round trips and special tokens, tokenizer/checkpoint compatibility, causal attention, SDPA parity, KV cache parity, RoPE validation, optimizer and scheduler behavior, generation sampling and EOS stopping, deterministic CPU resume, FastAPI validation, and the tiny raw-text-to-HTTP integration path.

Tokenizer artifacts are pickle files; load only artifacts you trust.

## Repository Structure

~~~text
cs336_basics/
  tokenizer.py       BPE tokenizer and artifact identity metadata
  train_bpe.py       BPE-training CLI
  prepare_data.py    raw-text-to-NPY CLI
  transformer.py     Transformer model and attention backends
  trainer.py         training loop and checkpoint orchestration
  service.py         FastAPI application
configs/
  tiny_cpu_smoke.yaml
tests/
Dockerfile
requirements-service.txt
~~~

## Acknowledgments

Built as part of the Stanford CS336 Assignment 1 learning exercise.
