import json

import numpy as np
import pytest

from cs336_basics.prepare_data import main, prepare_datasets
from cs336_basics.tokenizer import save_tokenizer_artifacts


SPECIAL_TOKEN = "<|endoftext|>"


def write_artifacts(tmp_path, include_special_token=True):
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    if include_special_token:
        vocab[256] = SPECIAL_TOKEN.encode("utf-8")
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    save_tokenizer_artifacts(vocab, [], vocab_path, merges_path)
    return vocab_path, merges_path


def test_prepare_datasets_is_deterministic_and_writes_metadata(tmp_path):
    vocab_path, merges_path = write_artifacts(tmp_path)
    train_text = tmp_path / "train.txt"
    validation_text = tmp_path / "validation.txt"
    train_text.write_text("hello <|endoftext|> world", encoding="utf-8")
    validation_text.write_text("world <|endoftext|> hello", encoding="utf-8")

    first = prepare_datasets(
        train_text_path=train_text,
        validation_text_path=validation_text,
        vocab_path=vocab_path,
        merges_path=merges_path,
        train_output_path=tmp_path / "first_train.npy",
        validation_output_path=tmp_path / "first_validation.npy",
        special_tokens=[SPECIAL_TOKEN],
    )
    second = prepare_datasets(
        train_text_path=train_text,
        validation_text_path=validation_text,
        vocab_path=vocab_path,
        merges_path=merges_path,
        train_output_path=tmp_path / "second_train.npy",
        validation_output_path=tmp_path / "second_validation.npy",
        special_tokens=[SPECIAL_TOKEN],
    )

    np.testing.assert_array_equal(
        np.load(tmp_path / "first_train.npy"),
        np.load(tmp_path / "second_train.npy"),
    )
    assert first["tokenizer"] == second["tokenizer"]
    assert first["datasets"]["train"]["sha256"] == second["datasets"]["train"]["sha256"]

    metadata_path = tmp_path / "metadata.json"
    main(
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
            str(tmp_path / "cli_train.npy"),
            "--validation-output-path",
            str(tmp_path / "cli_validation.npy"),
            "--metadata-path",
            str(metadata_path),
            "--special-token",
            SPECIAL_TOKEN,
        ]
    )
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["tokenizer"] == first["tokenizer"]


@pytest.mark.parametrize(
    ("train_text", "include_special_token", "expected_error"),
    [
        ("", True, "empty"),
        ("hello", False, "special token"),
    ],
)
def test_prepare_datasets_rejects_malformed_inputs(
    tmp_path, train_text, include_special_token, expected_error
):
    vocab_path, merges_path = write_artifacts(tmp_path, include_special_token)
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text(train_text, encoding="utf-8")
    validation_path.write_text("validation", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        prepare_datasets(
            train_text_path=train_path,
            validation_text_path=validation_path,
            vocab_path=vocab_path,
            merges_path=merges_path,
            train_output_path=tmp_path / "train.npy",
            validation_output_path=tmp_path / "validation.npy",
            special_tokens=[SPECIAL_TOKEN],
        )
