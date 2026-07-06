# Mini GPT from Scratch

A small GPT-style language model training pipeline implemented from scratch in PyTorch.

This project covers the core components of a modern language modeling system:

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

This repository implements:

* Byte-level BPE tokenizer training
* BPE encoding and decoding
* GPT-style decoder-only Transformer
* RMSNorm, RoPE, causal self-attention, and SwiGLU
* Cross-entropy next-token prediction loss
* AdamW optimizer
* Cosine learning-rate schedule with warmup
* Gradient clipping
* Validation and checkpointing
* Autoregressive text generation with temperature and top-p sampling

The goal of this project is to understand the full language modeling pipeline from raw text preprocessing to Transformer training and generation, without relying on high-level language modeling libraries.

## Tokenizer

The tokenizer is a byte-level BPE tokenizer.

It starts from the 256 raw byte vocabulary and learns frequent byte-sequence merges from the training corpus. This makes the tokenizer able to represent arbitrary text while still learning compact subword-like units.

Special tokens are handled as indivisible tokens during encoding and as hard boundaries during tokenizer training.

## Model

The language model is a decoder-only Transformer. Given a sequence of token IDs, it predicts the next token at each position.

The Transformer blocks use:

* Causal multi-head self-attention
* Rotary positional embeddings
* RMSNorm
* SwiGLU feed-forward layers
* Pre-norm Transformer architecture

## Training

The training loop samples token windows from tokenized datasets, computes next-token cross-entropy loss, and updates the model using AdamW.

The implementation supports:

* Random batch sampling from tokenized data
* Cosine learning-rate scheduling with warmup
* Gradient clipping
* Validation loss evaluation
* Checkpoint save/load

## Generation

The model supports autoregressive text generation with:

* Temperature scaling
* Top-p / nucleus sampling
* Optional early stopping on special tokens

## Results / Example Training Run

I trained a small Transformer language model on TinyStories using a 10K byte-level BPE vocabulary.

Example setup:

| Setting             |                        Value |
| ------------------- | ---------------------------: |
| Dataset             |                  TinyStories |
| Vocabulary size     |                       10,000 |
| Batch size          |                          128 |
| Learning rate       |                         1e-3 |
| Training iterations |                       10,000 |
| Decoding            | Temperature + top-p sampling |

Example prompt:

```text
Once upon a time
```

Example generated output:

```text
Once upon a time, there was a friendly girl named Lily...
```

The trained model learned to generate simple story-like text with short sentences, recurring characters, and TinyStories-style narrative structure. Since the model is small and trained with limited compute, it can still repeat phrases or lose long-range coherence.

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

* How byte-level BPE tokenization works
* How tokenized datasets are prepared for language modeling
* How decoder-only Transformers perform next-token prediction
* How optimizer, scheduler, checkpointing, and generation fit together in a full training pipeline
* How decoding parameters such as temperature and top-p affect generated text quality
