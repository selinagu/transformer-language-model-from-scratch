import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.config import ModelConfig, TrainingConfig
from cs336_basics.data import get_batch
from cs336_basics.optimizer import cross_entropy, get_lr_cosine_schedule, gradient_clipping


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: str,
        checkpoint_path: str | Path,
        checkpoint_metadata: dict[str, Any],
        log_path: str | Path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.valid_data = valid_data
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_metadata = checkpoint_metadata
        self.log_path = Path(log_path)

    def _set_learning_rate(self, step: int) -> float:
        lr = get_lr_cosine_schedule(
            step,
            self.training_config.learning_rate,
            self.training_config.min_learning_rate,
            self.training_config.warmup_iters,
            self.training_config.max_iters,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def train_step(self) -> float:
        self.model.train()
        inputs, targets = get_batch(
            self.train_data,
            self.training_config.batch_size,
            self.model_config.context_length,
            self.device,
        )
        self.optimizer.zero_grad()
        loss = cross_entropy(self.model(inputs), targets)
        loss.backward()

        if self.training_config.grad_clip is not None:
            gradient_clipping(self.model.parameters(), self.training_config.grad_clip)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        was_training = self.model.training
        self.model.eval()
        losses = []

        for _ in range(self.training_config.val_batches):
            inputs, targets = get_batch(
                self.valid_data,
                self.training_config.batch_size,
                self.model_config.context_length,
                self.device,
            )
            losses.append(cross_entropy(self.model(inputs), targets).item())

        self.model.train(was_training)
        return sum(losses) / len(losses)

    def resume(self, checkpoint_path: str | Path) -> int:
        return load_checkpoint(checkpoint_path, self.model, self.optimizer)

    def save_checkpoint(self, next_step: int) -> None:
        save_checkpoint(
            self.model,
            self.optimizer,
            next_step,
            self.checkpoint_path,
            metadata=self.checkpoint_metadata,
        )

    def _write_log_header(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="") as handle:
            csv.DictWriter(
                handle,
                fieldnames=["step", "train_loss", "val_loss", "lr", "elapsed_seconds"],
            ).writeheader()

    def _write_log_row(
        self,
        *,
        step: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        elapsed_seconds: float,
    ) -> None:
        with self.log_path.open("a", newline="") as handle:
            csv.DictWriter(
                handle,
                fieldnames=["step", "train_loss", "val_loss", "lr", "elapsed_seconds"],
            ).writerow(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "elapsed_seconds": elapsed_seconds,
                }
            )

    def fit(self, *, start_step: int = 0) -> None:
        if start_step < 0 or start_step > self.training_config.max_iters:
            raise ValueError("start_step must be between 0 and max_iters")
        if start_step == 0:
            self._write_log_header()

        started_at = time.time()
        for step in range(start_step, self.training_config.max_iters):
            lr = self._set_learning_rate(step)
            train_loss = self.train_step()
            next_step = step + 1

            if step % self.training_config.eval_interval == 0:
                val_loss = self.evaluate()
                elapsed_seconds = time.time() - started_at
                print(
                    f"step {step} | train loss {train_loss:.4f} | "
                    f"val loss {val_loss:.4f} | lr {lr:.2e} | "
                    f"time {elapsed_seconds:.1f}s"
                )
                self._write_log_row(
                    step=step,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=lr,
                    elapsed_seconds=elapsed_seconds,
                )

            if next_step % self.training_config.save_interval == 0:
                self.save_checkpoint(next_step)

        self.save_checkpoint(self.training_config.max_iters)
