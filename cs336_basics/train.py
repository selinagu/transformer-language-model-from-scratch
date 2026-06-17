import argparse
import time
import csv 
import json
import os
import numpy as np
import torch

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import (
    AdamW,
    get_batch,
    cross_entropy,
    gradient_clipping,
    get_lr_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
)

def estimate_val_loss(model, valid_data, args):
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(args.val_batches):
            xb, yb = get_batch(
                valid_data,
                args.batch_size,
                args.context_length,
                args.device,
            )
            logits = model(xb)
            loss = cross_entropy(logits, yb)
            losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--log_path", type=str, default="train_log.csv")

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=10000.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)

    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    with open(args.log_path + ".config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")

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

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
    else:
        start_iter = 0

    if start_iter == 0:
        with open(args.log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["steps", "train_loss", "val_loss", "lr", "elapsed_seconds"],
            )
            writer.writeheader()

    model.train()
    start_time = time.time()

    for step in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(
            step,
            args.lr,
            args.min_lr,
            args.warmup_iters,
            args.max_iters,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            args.device,
        )

        optimizer.zero_grad()
        logits = model(xb)
        loss = cross_entropy(logits, yb)
        loss.backward()

        if args.grad_clip is not None:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        if step % args.eval_interval == 0:
            val_loss = estimate_val_loss(model, valid_data, args)
            elapsed = time.time() - start_time
            print(
                f"step {step} | "
                f"train loss {loss.item():.4f} | "
                f"val loss {val_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"time {elapsed:.1f}s"
            )

            with open(args.log_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["step", "train_loss", "val_loss", "lr", "elapsed_seconds"],
                )
                writer.writerow(
                    {
                        "step": step,
                        "train_loss": loss.item(),
                        "val_loss": val_loss,
                        "lr": lr,
                        "elapsed_seconds": elapsed,
                    }
                )

        if step > 0 and step % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)

    save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)

if __name__ == "__main__":
    main()