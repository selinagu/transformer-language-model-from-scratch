import sys

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from cs336_basics import train
from cs336_basics.prepare_data import main as prepare_data_main
from cs336_basics.service import load_app
from cs336_basics.tokenizer import Tokenizer, save_tokenizer_artifacts
from cs336_basics.train_bpe import main as train_bpe_main


SPECIAL_TOKEN = "<|endoftext|>"


def test_tiny_cpu_pipeline_from_raw_text_to_http_service(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    train_text = tmp_path / "train.txt"
    validation_text = tmp_path / "validation.txt"
    train_text.write_text(
        ("a tiny story begins <|endoftext|> a tiny story continues\n" * 24),
        encoding="utf-8",
    )
    validation_text.write_text(
        ("a tiny validation story <|endoftext|> continues\n" * 12),
        encoding="utf-8",
    )
    vocab_path = artifacts / "vocab.pkl"
    merges_path = artifacts / "merges.pkl"
    train_bpe_main(
        [
            "--input-path",
            str(train_text),
            "--vocab-size",
            "264",
            "--vocab-path",
            str(vocab_path),
            "--merges-path",
            str(merges_path),
            "--special-token",
            SPECIAL_TOKEN,
            "--num-processes",
            "1",
        ]
    )

    train_tokens = artifacts / "train.npy"
    validation_tokens = artifacts / "validation.npy"
    prepare_data_main(
        [
            "--train-text-path",
            str(train_text),
            "--validation-text-path",
            str(validation_text),
            "--vocab-path",
            str(vocab_path),
            "--merges-path",
            str(merges_path),
            "--train-output-path",
            str(train_tokens),
            "--validation-output-path",
            str(validation_tokens),
            "--special-token",
            SPECIAL_TOKEN,
        ]
    )
    assert npy_token_count(train_tokens) > 4
    assert npy_token_count(validation_tokens) > 4

    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "vocab_size": 264,
                    "context_length": 4,
                    "d_model": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "d_ff": 16,
                },
                "training": {
                    "batch_size": 2,
                    "max_iters": 2,
                    "warmup_iters": 0,
                    "eval_interval": 100,
                    "save_interval": 1,
                    "seed": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = artifacts / "checkpoint.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--train_data",
            str(train_tokens),
            "--valid_data",
            str(validation_tokens),
            "--vocab_path",
            str(vocab_path),
            "--merges_path",
            str(merges_path),
            "--checkpoint_path",
            str(checkpoint_path),
            "--log_path",
            str(artifacts / "train.csv"),
            "--device",
            "cpu",
        ],
    )
    train.main()

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["metadata"]["tokenizer"]["vocab_size"] == 264

    client = TestClient(load_app(checkpoint_path, vocab_path, merges_path, device="cpu"))
    assert client.get("/health").status_code == 200
    response = client.post(
        "/generate",
        json={"prompt": "a tiny", "max_new_tokens": 2, "temperature": 0},
    )
    assert response.status_code == 200
    assert isinstance(response.json()["text"], str)

    tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    incompatible_vocab = dict(tokenizer.vocab)
    incompatible_vocab[0], incompatible_vocab[1] = incompatible_vocab[1], incompatible_vocab[0]
    incompatible_vocab_path = artifacts / "incompatible_vocab.pkl"
    save_tokenizer_artifacts(incompatible_vocab, [], incompatible_vocab_path, merges_path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_app(checkpoint_path, incompatible_vocab_path, merges_path, device="cpu")


def npy_token_count(path):
    import numpy as np

    return int(np.load(path).size)
