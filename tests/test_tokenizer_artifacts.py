import pytest

from cs336_basics.tokenizer import Tokenizer, save_tokenizer_artifacts
from cs336_basics.train_bpe import main, train_bpe


SPECIAL_TOKEN = "<|endoftext|>"


def test_saved_tokenizer_artifacts_round_trip(tmp_path):
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    vocab[256] = SPECIAL_TOKEN.encode("utf-8")
    vocab[257] = b"he"
    vocab[258] = b"hel"
    merges = [(b"h", b"e"), (b"he", b"l")]

    vocab_path = tmp_path / "nested" / "vocab.pkl"
    merges_path = tmp_path / "nested" / "merges.pkl"
    save_tokenizer_artifacts(vocab, merges, vocab_path, merges_path)

    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=[SPECIAL_TOKEN],
    )

    assert vocab_path.exists()
    assert merges_path.exists()
    assert tokenizer.vocab == vocab
    assert tokenizer.merges_ranks == {merge: rank for rank, merge in enumerate(merges)}
    assert tokenizer.decode(tokenizer.encode(f"hello{SPECIAL_TOKEN}world")) == (
        f"hello{SPECIAL_TOKEN}world"
    )


def test_train_bpe_cli_writes_loadable_artifacts(tmp_path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(
        f"hello world{SPECIAL_TOKEN}hello world{SPECIAL_TOKEN}",
        encoding="utf-8",
    )
    vocab_path = tmp_path / "artifacts" / "vocab.pkl"
    merges_path = tmp_path / "artifacts" / "merges.pkl"

    main(
        [
            "--input-path",
            str(input_path),
            "--vocab-size",
            "260",
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

    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=[SPECIAL_TOKEN],
    )

    assert len(tokenizer.vocab) == 260
    assert tokenizer.decode(tokenizer.encode(f"hello{SPECIAL_TOKEN}world")) == (
        f"hello{SPECIAL_TOKEN}world"
    )


def test_train_bpe_rejects_non_positive_process_count(tmp_path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="num_processes"):
        train_bpe(
            input_path=str(input_path),
            vocab_size=256,
            special_tokens=[],
            num_processes=0,
        )


def test_tokenizer_metadata_fingerprint_is_deterministic_and_identity_sensitive():
    from cs336_basics.tokenizer import tokenizer_artifact_metadata

    vocab = {
        0: b"a",
        1: b"b",
        2: SPECIAL_TOKEN.encode("utf-8"),
    }
    first = tokenizer_artifact_metadata(Tokenizer(vocab.copy(), [], special_tokens=[SPECIAL_TOKEN]))
    second = tokenizer_artifact_metadata(Tokenizer(vocab.copy(), [], special_tokens=[SPECIAL_TOKEN]))
    remapped = tokenizer_artifact_metadata(
        Tokenizer(
            {0: b"b", 1: b"a", 2: SPECIAL_TOKEN.encode("utf-8")},
            [],
            special_tokens=[SPECIAL_TOKEN],
        )
    )

    assert first == second
    assert first["special_tokens"] == {SPECIAL_TOKEN: 2}
    assert first["fingerprint"] != remapped["fingerprint"]
