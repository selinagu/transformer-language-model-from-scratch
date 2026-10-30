import argparse
import os
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from cs336_basics.config import ModelConfig
from cs336_basics.decoder import generate
from cs336_basics.tokenizer import Tokenizer, tokenizer_artifact_metadata
from cs336_basics.transformer import TransformerLM


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0)
    top_p: float | None = Field(default=0.9, gt=0, le=1)
    use_kv_cache: bool = True


class GenerationResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    device: str
    context_length: int
    attention_backend: str
    checkpoint_iteration: int | None = None


def validate_tokenizer_compatibility(
    model_config: ModelConfig,
    tokenizer: Tokenizer,
    expected_metadata: dict[str, Any],
) -> None:
    actual_metadata = tokenizer_artifact_metadata(tokenizer)
    if actual_metadata["vocab_size"] != model_config.vocab_size:
        raise ValueError(
            "tokenizer vocabulary size does not match checkpoint model_config: "
            f"{actual_metadata['vocab_size']} != {model_config.vocab_size}"
        )

    required_fields = {"format_version", "vocab_size", "special_tokens", "fingerprint"}
    missing_fields = required_fields - set(expected_metadata)
    if missing_fields:
        raise ValueError(
            "checkpoint tokenizer metadata is incomplete; missing "
            f"{sorted(missing_fields)}"
        )

    if expected_metadata["vocab_size"] != actual_metadata["vocab_size"]:
        raise ValueError(
            "tokenizer vocabulary size does not match checkpoint metadata: "
            f"{actual_metadata['vocab_size']} != {expected_metadata['vocab_size']}"
        )
    if expected_metadata["special_tokens"] != actual_metadata["special_tokens"]:
        raise ValueError("tokenizer special-token mapping does not match checkpoint metadata")
    if expected_metadata["fingerprint"] != actual_metadata["fingerprint"]:
        raise ValueError("tokenizer fingerprint does not match checkpoint metadata")


def create_app(
    model: TransformerLM,
    tokenizer: Tokenizer,
    *,
    device: str,
    eos_token_id: int | None = None,
    checkpoint_iteration: int | None = None,
) -> FastAPI:
    app = FastAPI(title="Mini GPT API", version="1.0.0")
    generation_lock = Lock()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            device=device,
            context_length=model.context_length,
            attention_backend=model.attention_backend,
            checkpoint_iteration=checkpoint_iteration,
        )

    @app.post("/generate", response_model=GenerationResponse)
    def generate_text(request: GenerationRequest) -> GenerationResponse:
        with generation_lock:
            text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                eos_token_id=eos_token_id,
                temperature=request.temperature,
                top_p=request.top_p,
                device=device,
                use_kv_cache=request.use_kv_cache,
            )
        return GenerationResponse(text=text)

    return app


def load_app(
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    merges_path: str | Path,
    *,
    device: str,
) -> FastAPI:
    for label, path in (("checkpoint", checkpoint_path), ("vocabulary", vocab_path), ("merges", merges_path)):
        if not Path(path).is_file():
            raise ValueError(
                f"service artifact is missing or not a file: {label} ({path})"
            )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata: dict[str, Any] = checkpoint.get("metadata", {})
    model_config_data = metadata.get("model_config")
    if model_config_data is None:
        raise ValueError(
            "service deployment requires checkpoint metadata.model_config. "
            "Re-save this legacy checkpoint with the current training pipeline."
        )
    tokenizer_metadata = metadata.get("tokenizer")
    if tokenizer_metadata is None:
        raise ValueError(
            "service deployment cannot verify a legacy checkpoint without tokenizer metadata. "
            "Re-train or re-save it with the current training pipeline."
        )

    model_config = ModelConfig(**model_config_data)
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=["<|endoftext|>"],
    )
    validate_tokenizer_compatibility(model_config, tokenizer, tokenizer_metadata)

    model = TransformerLM(**model_config.model_kwargs(), device=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    eos_token_id = tokenizer.bytes_to_id.get(b"<|endoftext|>")

    return create_app(
        model,
        tokenizer,
        device=device,
        eos_token_id=eos_token_id,
        checkpoint_iteration=checkpoint.get("iteration"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Mini GPT checkpoint over HTTP.")
    parser.add_argument("--checkpoint-path", default=os.getenv("CHECKPOINT_PATH"))
    parser.add_argument("--vocab-path", default=os.getenv("VOCAB_PATH"))
    parser.add_argument("--merges-path", default=os.getenv("MERGES_PATH"))
    parser.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()

    missing = [
        name
        for name, value in {
            "checkpoint-path": args.checkpoint_path,
            "vocab-path": args.vocab_path,
            "merges-path": args.merges_path,
        }.items()
        if not value
    ]
    if missing:
        parser.error(f"Provide --{', --'.join(missing)} or matching environment variables.")
    return args


def main() -> None:
    args = parse_args()
    app = load_app(
        args.checkpoint_path,
        args.vocab_path,
        args.merges_path,
        device=args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
