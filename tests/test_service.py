import torch
import pytest
from fastapi.testclient import TestClient

from cs336_basics.service import create_app, load_app
from cs336_basics.tokenizer import (
    Tokenizer,
    save_tokenizer_artifacts,
    tokenizer_artifact_metadata,
)
from cs336_basics.transformer import TransformerLM


SPECIAL_TOKEN = "<|endoftext|>"


class IntegerTokenizer:
    def encode(self, prompt: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(map(str, token_ids))


def make_model(vocab_size: int = 8) -> TransformerLM:
    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=4,
        d_ff=32,
    ).eval()


def make_client() -> TestClient:
    app = create_app(
        make_model(vocab_size=32),
        IntegerTokenizer(),
        device="cpu",
        checkpoint_iteration=42,
    )
    return TestClient(app)


def tokenizer_vocab() -> dict[int, bytes]:
    return {
        0: b"a",
        1: b"b",
        2: b"c",
        3: b"d",
        4: b"e",
        5: b"f",
        6: b"g",
        7: SPECIAL_TOKEN.encode("utf-8"),
    }


def write_artifacts(tmp_path, vocab: dict[int, bytes]) -> tuple:
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    save_tokenizer_artifacts(vocab, [], vocab_path, merges_path)
    return vocab_path, merges_path


def write_checkpoint(
    tmp_path,
    model: TransformerLM,
    tokenizer_metadata: dict | None,
):
    metadata = {"model_config": {
        "vocab_size": model.vocab_size,
        "context_length": model.context_length,
        "d_model": model.d_model,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "theta": model.theta,
        "attention_backend": model.attention_backend,
    }}
    if tokenizer_metadata is not None:
        metadata["tokenizer"] = tokenizer_metadata

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "iteration": 7, "metadata": metadata},
        checkpoint_path,
    )
    return checkpoint_path


def expected_tokenizer_metadata(vocab: dict[int, bytes]) -> dict:
    tokenizer = Tokenizer(vocab.copy(), [], special_tokens=[SPECIAL_TOKEN])
    return tokenizer_artifact_metadata(tokenizer)


def test_health_reports_loaded_model_metadata():
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "device": "cpu",
        "context_length": 8,
        "attention_backend": "naive",
        "checkpoint_iteration": 42,
    }


def test_generate_returns_text():
    response = make_client().post(
        "/generate",
        json={"prompt": "hello", "max_new_tokens": 2, "temperature": 0},
    )

    assert response.status_code == 200
    assert response.json()["text"].split()[:3] == ["1", "2", "3"]


def test_generate_rejects_invalid_sampling_parameters():
    response = make_client().post(
        "/generate",
        json={"prompt": "hello", "top_p": 1.5},
    )

    assert response.status_code == 422


def test_load_app_accepts_matching_tokenizer_artifacts(tmp_path):
    model = make_model()
    vocab = tokenizer_vocab()
    vocab_path, merges_path = write_artifacts(tmp_path, vocab)
    checkpoint_path = write_checkpoint(
        tmp_path,
        model,
        expected_tokenizer_metadata(vocab),
    )

    response = TestClient(
        load_app(checkpoint_path, vocab_path, merges_path, device="cpu")
    ).get("/health")

    assert response.status_code == 200
    assert response.json()["checkpoint_iteration"] == 7


def test_load_app_rejects_different_tokenizer_vocabulary_size(tmp_path):
    model = make_model()
    expected_vocab = tokenizer_vocab()
    actual_vocab = {**expected_vocab, 8: b"z"}
    vocab_path, merges_path = write_artifacts(tmp_path, actual_vocab)
    checkpoint_path = write_checkpoint(
        tmp_path,
        model,
        expected_tokenizer_metadata(expected_vocab),
    )

    with pytest.raises(ValueError, match="vocabulary size"):
        load_app(checkpoint_path, vocab_path, merges_path, device="cpu")


def test_load_app_rejects_same_size_different_token_id_mapping(tmp_path):
    model = make_model()
    expected_vocab = tokenizer_vocab()
    actual_vocab = {
        0: b"c",
        1: b"b",
        2: b"a",
        3: b"d",
        4: b"e",
        5: b"f",
        6: b"g",
        7: SPECIAL_TOKEN.encode("utf-8"),
    }
    vocab_path, merges_path = write_artifacts(tmp_path, actual_vocab)
    checkpoint_path = write_checkpoint(
        tmp_path,
        model,
        expected_tokenizer_metadata(expected_vocab),
    )

    with pytest.raises(ValueError, match="fingerprint"):
        load_app(checkpoint_path, vocab_path, merges_path, device="cpu")


def test_load_app_rejects_legacy_checkpoint_without_tokenizer_metadata(tmp_path):
    model = make_model()
    vocab = tokenizer_vocab()
    vocab_path, merges_path = write_artifacts(tmp_path, vocab)
    checkpoint_path = write_checkpoint(tmp_path, model, tokenizer_metadata=None)

    with pytest.raises(ValueError, match="legacy checkpoint"):
        load_app(checkpoint_path, vocab_path, merges_path, device="cpu")


def test_load_app_rejects_different_special_token_mapping(tmp_path):
    model = make_model()
    vocab = tokenizer_vocab()
    vocab_path, merges_path = write_artifacts(tmp_path, vocab)
    metadata = expected_tokenizer_metadata(vocab)
    metadata["special_tokens"] = {SPECIAL_TOKEN: 0}
    checkpoint_path = write_checkpoint(tmp_path, model, metadata)

    with pytest.raises(ValueError, match="special-token mapping"):
        load_app(checkpoint_path, vocab_path, merges_path, device="cpu")


def test_load_app_rejects_missing_artifacts_with_clear_error(tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(ValueError, match="service artifact is missing"):
        load_app(missing, missing, missing, device="cpu")
