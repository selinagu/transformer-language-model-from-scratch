import argparse

import torch

from benchmarks.common import benchmark, write_csv
from cs336_basics.config import ModelConfig
from cs336_basics.transformer import TransformerLM


@torch.inference_mode()
def generate_without_cache(
    model: TransformerLM, prompt_ids: torch.Tensor, max_new_tokens: int
) -> torch.Tensor:
    token_ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(token_ids)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        token_ids = torch.cat((token_ids, next_token), dim=1)
    return token_ids


@torch.inference_mode()
def generate_with_cache(
    model: TransformerLM, prompt_ids: torch.Tensor, max_new_tokens: int
) -> torch.Tensor:
    token_ids = prompt_ids.clone()
    logits, past_key_values = model(token_ids, use_cache=True)

    for step in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        token_ids = torch.cat((token_ids, next_token), dim=1)
        if step + 1 < max_new_tokens:
            logits, past_key_values = model(
                next_token, past_key_values=past_key_values, use_cache=True
            )
    return token_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark autoregressive decoding.")
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=352)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="benchmarks/results/generation.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    context_length = args.prompt_length + args.max_new_tokens
    config = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )
    device = torch.device(args.device)
    model = TransformerLM(**config.model_kwargs(), device=device).eval()
    prompt_ids = torch.randint(
        config.vocab_size,
        (args.batch_size, args.prompt_length),
        device=device,
    )

    @torch.inference_mode()
    def prefill():
        model(prompt_ids)

    @torch.inference_mode()
    def decode_full_prefix():
        generate_without_cache(model, prompt_ids, args.max_new_tokens)

    @torch.inference_mode()
    def decode_kv_cache():
        generate_with_cache(model, prompt_ids, args.max_new_tokens)

    common_details = {
        "backend": model.attention_backend,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "context_length": context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
    }
    results = [
        benchmark(
            prefill,
            name="generation_prefill",
            device=device,
            dtype=prompt_ids.dtype,
            warmup_iterations=args.warmup,
            measurement_iterations=args.iterations,
            details={
                **common_details,
                "work_items_per_iteration": args.batch_size * args.prompt_length,
            },
        ),
        benchmark(
            decode_full_prefix,
            name="generation_decode_full_prefix",
            device=device,
            dtype=prompt_ids.dtype,
            warmup_iterations=args.warmup,
            measurement_iterations=args.iterations,
            details={
                **common_details,
                "decode_strategy": "full_prefix",
                "work_items_per_iteration": args.batch_size * args.max_new_tokens,
            },
        ),
        benchmark(
            decode_kv_cache,
            name="generation_decode_kv_cache",
            device=device,
            dtype=prompt_ids.dtype,
            warmup_iterations=args.warmup,
            measurement_iterations=args.iterations,
            details={
                **common_details,
                "decode_strategy": "kv_cache",
                "work_items_per_iteration": args.batch_size * args.max_new_tokens,
            },
        ),
    ]

    write_csv(args.output, results)
    for result in results:
        print(
            f"{result.name} median_ms={result.median_latency_ms:.3f} "
            f"tokens_per_second={result.throughput_per_second:.1f}"
        )


if __name__ == "__main__":
    main()
