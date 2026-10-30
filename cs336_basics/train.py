import argparse
import random
import os
from pathlib import Path

import numpy as np
import torch

from cs336_basics.config import (
    ExperimentConfig,
    apply_overrides,
    load_experiment_config,
    save_experiment_config,
)
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import Tokenizer, tokenizer_artifact_metadata
from cs336_basics.trainer import Trainer
from cs336_basics.transformer import TransformerLM


MODEL_CONFIG_FIELDS = (
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "theta",
)
TRAINING_CONFIG_FIELDS = (
    "batch_size",
    "max_iters",
    "learning_rate",
    "min_learning_rate",
    "warmup_iters",
    "weight_decay",
    "grad_clip",
    "eval_interval",
    "val_batches",
    "save_interval",
    "seed",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--special_token", action="append", default=None)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--log_path", type=str, default="train_log.csv")
    parser.add_argument("--device", type=str, default="cuda:0")

    for field in MODEL_CONFIG_FIELDS:
        parser.add_argument(f"--{field}", type=float if field == "theta" else int, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--min_lr", dest="min_learning_rate", type=float, default=None)
    parser.add_argument("--warmup_iters", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--val_batches", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_config = load_experiment_config(args.config) if args.config else ExperimentConfig()
    experiment_config = apply_overrides(
        experiment_config,
        model_overrides={field: getattr(args, field) for field in MODEL_CONFIG_FIELDS},
        training_overrides={
            field: getattr(args, field) for field in TRAINING_CONFIG_FIELDS
        },
    )
    model_config = experiment_config.model
    training_config = experiment_config.training
    special_tokens = args.special_token or ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=special_tokens,
    )
    tokenizer_metadata = tokenizer_artifact_metadata(tokenizer)
    if tokenizer_metadata["vocab_size"] != model_config.vocab_size:
        raise ValueError(
            "tokenizer vocabulary size does not match model vocab_size: "
            f"{tokenizer_metadata['vocab_size']} != {model_config.vocab_size}"
        )

    torch.manual_seed(training_config.seed)
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
    save_experiment_config(experiment_config, Path(args.log_path).with_suffix(".resolved.yaml"))

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")
    model = TransformerLM(**model_config.model_kwargs(), device=args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=training_config.weight_decay,
    )
    checkpoint_metadata = {
        "format_version": 2,
        "checkpoint_step_semantics": "next_step",
        "model_config": model_config.model_kwargs(),
        "tokenizer": tokenizer_metadata,
        "training_config": experiment_config.to_dict()["training"],
    }
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        valid_data=valid_data,
        model_config=model_config,
        training_config=training_config,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        log_path=args.log_path,
    )
    start_step = trainer.resume(args.resume_from) if args.resume_from else 0
    trainer.fit(start_step=start_step)


if __name__ == "__main__":
    main()
