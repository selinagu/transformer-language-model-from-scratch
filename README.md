# Mini GPT from Scratch

This repository implements a small GPT-style language model training pipeline from scratch in PyTorch.

The goal is to understand the full language modeling pipeline:

```text
raw text
→ byte-level BPE tokenizer
→ tokenized dataset
→ decoder-only Transformer LM
→ training loop
→ checkpointing
→ text generation
```

## Overview

The implementation includes:

* Byte-level BPE tokenizer training
* BPE encoding and decoding
* GPT-style decoder-only Transformer
* RMSNorm, RoPE, causal self-attention, and SwiGLU
* Cross-entropy next-token prediction loss
* AdamW optimizer
* Cosine learning-rate schedule with warmup
* Gradient clipping
* Validation and checkpointing
* Autoregressive generation with temperature and top-p sampling

## Tokenizer

The tokenizer is a byte-level BPE tokenizer. It starts from the 256 byte vocabulary and learns frequent byte-sequence merges from a training corpus.

Special tokens are handled as indivisible tokens during encoding and as hard boundaries during tokenizer training.

## Model

The model is a decoder-only Transformer language model. Given a sequence of token IDs, it predicts the next token at each position.

The Transformer blocks use causal multi-head self-attention, rotary positional embeddings, RMSNorm, and SwiGLU feed-forward layers.

## Training

The training loop samples token windows from tokenized datasets, computes next-token cross-entropy loss, and updates the model using AdamW.

It supports learning-rate scheduling, gradient clipping, validation loss evaluation, and checkpoint save/load.

## Generation

The model supports autoregressive text generation using temperature scaling and top-p sampling.

Example prompt:

```text
Once upon a time
```

Example output:

```text
Once upon a time, there was a friendly girl named Lily...
```

## Repository Structure

```text
.
├── cs336_basics/     # Core implementation
├── README.md
├── pyproject.toml
└── uv.lock
```

## What I Learned

Through this project, I gained a better understanding of:

* How BPE tokenization works
* How tokenized datasets are prepared for language modeling
* How decoder-only Transformers perform next-token prediction
* How optimizer, scheduler, checkpointing, and generation fit together in a full training pipeline
* How decoding parameters affect generated text quality
