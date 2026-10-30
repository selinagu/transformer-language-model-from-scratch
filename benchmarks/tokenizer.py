import argparse
from pathlib import Path

import torch

from benchmarks.common import benchmark, write_csv
from cs336_basics.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark BPE encoding throughput.")
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--merges-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", default="benchmarks/results/tokenizer.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    text = Path(args.input_path).read_text(encoding="utf-8")
    if args.max_chars is not None:
        text = text[: args.max_chars]
    if not text:
        raise ValueError("input text must not be empty")

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path)
    token_count = len(tokenizer.encode(text))
    input_bytes = len(text.encode("utf-8"))

    result = benchmark(
        lambda: tokenizer.encode(text),
        name="tokenizer_encode",
        device="cpu",
        dtype=torch.int64,
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations,
        details={
            "input_bytes": input_bytes,
            "token_count": token_count,
            "bytes_per_token": input_bytes / token_count,
            "work_items_per_iteration": input_bytes,
        },
    )
    write_csv(args.output, [result])
    print(
        f"median_ms={result.median_latency_ms:.3f} "
        f"bytes_per_second={result.throughput_per_second:.1f} "
        f"bytes_per_token={input_bytes / token_count:.3f}"
    )


if __name__ == "__main__":
    main()
