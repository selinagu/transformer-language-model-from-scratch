import argparse
import torch

from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoder import generate
from cs336_basics.config import ModelConfig

MODEL_CONFIG_FIELDS = (
    "vocab_size",
    "context_length",
    "num_layers",
    "d_model",
    "num_heads",
    "d_ff",
    "theta",
    "attention_backend",
)

def resolve_model_config(
    args: argparse.Namespace,
    checkpoint: dict,
) -> ModelConfig:
    metadata = checkpoint.get("metadata", {})
    metadata_config_data = metadata.get("model_config")

    if metadata_config_data is not None:
        return ModelConfig(**metadata_config_data)

    cli_values = {
        field: getattr(args, field, "naive" if field == "attention_backend" else None) for field in MODEL_CONFIG_FIELDS
    }

    missing = [
        field for field, value in cli_values.items() if value is None
    ]

    if missing:
        raise ValueError(
            "Legacy checkpoint has no model_config metadata. "
            f"Provide: {', '.join(missing)}"
        )

    return ModelConfig(**cli_values)


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

    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--attention_backend", choices=["naive", "sdpa"], default="naive")

    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=["<|endoftext|>"],
    )
    eos_token_id = tokenizer.bytes_to_id[b"<|endoftext|>"]

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model_config = resolve_model_config(args, checkpoint)

    model = TransformerLM(
        **model_config.model_kwargs(),
        device=args.device,
    )

    model.load_state_dict(checkpoint["model"])

    iteration = checkpoint["iteration"]

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