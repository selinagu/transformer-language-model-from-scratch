import argparse

import torch

from benchmarks.common import benchmark, write_csv
from cs336_basics.transformer import MultiHeadSelfAttention


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark an attention backend.")
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--backend", choices=["naive", "sdpa"], default="naive")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="benchmarks/results/attention.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if any(context <= 0 for context in args.contexts):
        raise ValueError("all context lengths must be positive")

    device = torch.device(args.device)
    attention = MultiHeadSelfAttention(
        d_model=args.d_model,
        num_heads=args.num_heads,
        max_seq_len=max(args.contexts),
        use_rope=True,
        attention_backend=args.backend,
        device=device,
    ).eval()
    results = []

    for context_length in args.contexts:
        inputs = torch.randn(args.batch_size, context_length, args.d_model, device=device)

        @torch.inference_mode()
        def operation():
            attention(inputs)

        results.append(
            benchmark(
                operation,
                name="attention_forward",
                device=device,
                dtype=inputs.dtype,
                warmup_iterations=args.warmup,
                measurement_iterations=args.iterations,
                details={
                    "backend": args.backend,
                    "batch_size": args.batch_size,
                    "context_length": context_length,
                    "d_model": args.d_model,
                    "num_heads": args.num_heads,
                    "work_items_per_iteration": args.batch_size * context_length,
                },
            )
        )

    write_csv(args.output, results)
    for result in results:
        print(
            f"context={result.details['context_length']} "
            f"median_ms={result.median_latency_ms:.3f} "
            f"tokens_per_second={result.throughput_per_second:.1f}"
        )


if __name__ == "__main__":
    main()
