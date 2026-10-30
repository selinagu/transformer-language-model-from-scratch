import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer, tokenizer_artifact_metadata


def _read_text(path: str | Path, label: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{label} text path is not a file: {source}")
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} text must be valid UTF-8: {source}") from error
    if not text:
        raise ValueError(f"{label} text file is empty: {source}")
    return text


def _load_tokenizer(
    vocab_path: str | Path,
    merges_path: str | Path,
    special_tokens: list[str],
) -> Tokenizer:
    for label, path in (("vocabulary", vocab_path), ("merges", merges_path)):
        if not Path(path).is_file():
            raise ValueError(f"tokenizer {label} artifact is not a file: {path}")

    tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    for token in special_tokens:
        if token.encode("utf-8") not in tokenizer.bytes_to_id:
            raise ValueError(f"special token is missing from tokenizer artifacts: {token}")
    tokenizer.special_tokens = list(special_tokens)
    return tokenizer


def _encode_text(tokenizer: Tokenizer, text: str, label: str) -> np.ndarray:
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise ValueError(f"{label} text produced no tokens")
    if min(token_ids) < 0 or max(token_ids) > np.iinfo(np.uint32).max:
        raise ValueError(f"{label} token IDs cannot be represented as uint32")
    return np.asarray(token_ids, dtype=np.uint32)


def _dataset_metadata(path: Path, tokens: np.ndarray) -> dict[str, str | int]:
    return {
        "path": str(path),
        "token_count": int(tokens.size),
        "dtype": str(tokens.dtype),
        "sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
    }


def prepare_datasets(
    *,
    train_text_path: str | Path,
    validation_text_path: str | Path,
    vocab_path: str | Path,
    merges_path: str | Path,
    train_output_path: str | Path,
    validation_output_path: str | Path,
    special_tokens: list[str],
) -> dict[str, object]:
    train_output = Path(train_output_path)
    validation_output = Path(validation_output_path)
    if train_output == validation_output:
        raise ValueError("train and validation output paths must be different")

    train_text = _read_text(train_text_path, "training")
    validation_text = _read_text(validation_text_path, "validation")
    tokenizer = _load_tokenizer(vocab_path, merges_path, special_tokens)
    train_tokens = _encode_text(tokenizer, train_text, "training")
    validation_tokens = _encode_text(tokenizer, validation_text, "validation")

    for output in (train_output, validation_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    np.save(train_output, train_tokens)
    np.save(validation_output, validation_tokens)

    return {
        "format_version": 1,
        "tokenizer": tokenizer_artifact_metadata(tokenizer),
        "datasets": {
            "train": _dataset_metadata(train_output, train_tokens),
            "validation": _dataset_metadata(validation_output, validation_tokens),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize raw UTF-8 text files into NumPy language-model datasets."
    )
    parser.add_argument("--train-text-path", required=True)
    parser.add_argument("--validation-text-path", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--merges-path", required=True)
    parser.add_argument("--train-output-path", required=True)
    parser.add_argument("--validation-output-path", required=True)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--special-token", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = prepare_datasets(
        train_text_path=args.train_text_path,
        validation_text_path=args.validation_text_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        train_output_path=args.train_output_path,
        validation_output_path=args.validation_output_path,
        special_tokens=args.special_token,
    )
    metadata_path = Path(args.metadata_path or f"{args.train_output_path}.metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
