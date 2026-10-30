import sys

import numpy as np
import torch
import yaml

from cs336_basics import train
from cs336_basics.tokenizer import (
    Tokenizer,
    save_tokenizer_artifacts,
    tokenizer_artifact_metadata,
)


def test_training_checkpoint_contains_tokenizer_metadata(tmp_path, monkeypatch):
    special_token = "<|endoftext|>"
    vocab = {token_id: bytes([97 + token_id]) for token_id in range(7)}
    vocab[7] = special_token.encode("utf-8")
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    save_tokenizer_artifacts(vocab, [], vocab_path, merges_path)

    tokens = np.arange(32, dtype=np.uint16) % 8
    train_path = tmp_path / "train.npy"
    valid_path = tmp_path / "valid.npy"
    np.save(train_path, tokens)
    np.save(valid_path, tokens)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "vocab_size": 8,
                    "context_length": 4,
                    "d_model": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "d_ff": 16,
                },
                "training": {
                    "batch_size": 2,
                    "max_iters": 1,
                    "warmup_iters": 0,
                    "eval_interval": 100,
                    "save_interval": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--train_data",
            str(train_path),
            "--valid_data",
            str(valid_path),
            "--vocab_path",
            str(vocab_path),
            "--merges_path",
            str(merges_path),
            "--checkpoint_path",
            str(checkpoint_path),
            "--log_path",
            str(tmp_path / "log.csv"),
            "--device",
            "cpu",
        ],
    )

    train.main()

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    expected = tokenizer_artifact_metadata(
        Tokenizer(vocab.copy(), [], special_tokens=[special_token])
    )
    assert checkpoint["metadata"]["tokenizer"] == expected
