import argparse

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW, load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoder import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--merges_path", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=10000.0)

    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=["<|endoftext|>"],
    )
    eos_token_id = tokenizer.bytes_to_id[b"<|endoftext|>"]

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=args.device,
    )

    optimizer = AdamW(model.parameters(), lr=1e-3)
    iteration = load_checkpoint(args.checkpoint_path, model, optimizer)

    print(f"Loaded checkpoint from iteration {iteration}\n")

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    print(text)


if __name__ == "__main__":
    main()