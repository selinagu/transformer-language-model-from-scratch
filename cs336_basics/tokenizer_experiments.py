import os
import time
import random
import numpy as np

from cs336_basics.tokenizer import Tokenizer


SPECIAL_TOKENS = ["<|endoftext|>"]


def load_tokenizer(vocab_path: str, merges_path: str) -> Tokenizer:
    return Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=SPECIAL_TOKENS,
    )


def split_documents(text: str, delimiter: str = "<|endoftext|>") -> list[str]:
    """
    Split a raw dataset into documents.
    """
    parts = text.split(delimiter)
    docs = []

    for part in parts:
        if part:
            docs.append(part + delimiter)

    return docs


def sample_documents(path: str, n: int = 10, seed: int = 0) -> list[str]:
    random.seed(seed)

    with open(path, encoding="utf-8") as f:
        text = f.read()

    docs = split_documents(text)
    if len(docs) < n:
        raise ValueError(f"Only found {len(docs)} documents in {path}")

    return random.sample(docs, n)


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    """
    Bytes per token.
    """
    total_bytes = 0
    total_tokens = 0

    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))

    return total_bytes / total_tokens, total_bytes, total_tokens


def throughput(tokenizer: Tokenizer, docs: list[str], repeats: int = 3) -> float:
    """
    Return bytes per second.
    """
    text = "".join(docs)
    n_bytes = len(text.encode("utf-8"))

    best_time = float("inf")

    for _ in range(repeats):
        start = time.perf_counter()
        _ = tokenizer.encode(text)
        end = time.perf_counter()
        best_time = min(best_time, end - start)

    return n_bytes / best_time


def encode_to_uint16(tokenizer: Tokenizer, input_path: str, output_path: str):
    """
    Encode a full dataset and save token ids as uint16.
    """
    ids = []

    with open(input_path, encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)

    arr = np.array(ids, dtype=np.uint16)
    np.save(output_path, arr)

    print(f"Saved {len(arr):,} tokens to {output_path}")


def main():
    tinystories_train_path = "data/TinyStoriesV2-GPT4-train.txt"
    tinystories_valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    openwebtext_train_path = "data/owt_train.txt"
    openwebtext_valid_path = "data/owt_valid.txt"

    tinystories_vocab_path = "outputs/tinystories_vocab_10000.pkl"
    tinystories_merges_path = "outputs/tinystories_merges_10000.pkl"

    openwebtext_vocab_path = "outputs/openwebtext_vocab_32000.pkl"
    openwebtext_merges_path = "outputs/openwebtext_merges_32000.pkl"

    ts_tokenizer = load_tokenizer(tinystories_vocab_path, tinystories_merges_path)
    owt_tokenizer = load_tokenizer(openwebtext_vocab_path, openwebtext_merges_path)

    # Sample 10 documents.
    ts_docs = sample_documents(tinystories_train_path, n=10, seed=0)
    owt_docs = sample_documents(openwebtext_train_path, n=10, seed=0)

    # Part (a)
    ts_ratio, ts_bytes, ts_tokens = compression_ratio(ts_tokenizer, ts_docs)
    owt_ratio, owt_bytes, owt_tokens = compression_ratio(owt_tokenizer, owt_docs)

    print("Part (a)")
    print(f"TinyStories tokenizer on TinyStories: {ts_ratio:.3f} bytes/token "
          f"({ts_bytes:,} bytes / {ts_tokens:,} tokens)")
    print(f"OpenWebText tokenizer on OpenWebText: {owt_ratio:.3f} bytes/token "
          f"({owt_bytes:,} bytes / {owt_tokens:,} tokens)")

    # Part (b)
    ts_on_owt_ratio, ts_on_owt_bytes, ts_on_owt_tokens = compression_ratio(ts_tokenizer, owt_docs)

    print("\nPart (b)")
    print(f"TinyStories tokenizer on OpenWebText: {ts_on_owt_ratio:.3f} bytes/token "
          f"({ts_on_owt_bytes:,} bytes / {ts_on_owt_tokens:,} tokens)")
    print(f"OpenWebText tokenizer on OpenWebText: {owt_ratio:.3f} bytes/token")

    # Part (c)
    # Use OpenWebText sample for throughput, since it is closer to general text.
    bytes_per_sec = throughput(owt_tokenizer, owt_docs, repeats=3)
    pile_bytes = 825 * 1024**3
    seconds = pile_bytes / bytes_per_sec
    hours = seconds / 3600

    print("\nPart (c)")
    print(f"Throughput: {bytes_per_sec:,.0f} bytes/sec")
    print(f"Estimated time for 825GB: {hours:.2f} hours")

    # Part (d)
    
    os.makedirs("data/tokenized", exist_ok=True)
    
    encode_to_uint16(
        ts_tokenizer,
        tinystories_train_path,
        "data/tokenized/tinystories_train_uint16.npy",
    )
    encode_to_uint16(
        ts_tokenizer,
        tinystories_valid_path,
        "data/tokenized/tinystories_valid_uint16.npy",
    )
    encode_to_uint16(
        owt_tokenizer,
        openwebtext_train_path,
        "data/tokenized/openwebtext_train_uint16.npy",
    )
    encode_to_uint16(
        owt_tokenizer,
        openwebtext_valid_path,
        "data/tokenized/openwebtext_valid_uint16.npy",
    )


if __name__ == "__main__":
    main()