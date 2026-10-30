from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    device: str
    dtype: str
    warmup_iterations: int
    measurement_iterations: int
    median_latency_ms: float
    mean_latency_ms: float
    details: dict[str, Any]

    @property
    def throughput_per_second(self) -> float | None:
        work_items = self.details.get("work_items_per_iteration")
        if work_items is None:
            return None
        return work_items / (self.median_latency_ms / 1_000)

    def to_row(self) -> dict[str, str | float | int]:
        return {
            "name": self.name,
            "device": self.device,
            "dtype": self.dtype,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "git_commit": current_git_commit(),
            "warmup_iterations": self.warmup_iterations,
            "measurement_iterations": self.measurement_iterations,
            "median_latency_ms": self.median_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "throughput_per_second": self.throughput_per_second or "",
            "details_json": json.dumps(self.details, sort_keys=True),
        }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    operation: Callable[[], Any],
    *,
    name: str,
    device: str | torch.device,
    dtype: torch.dtype,
    warmup_iterations: int,
    measurement_iterations: int,
    details: dict[str, Any],
) -> BenchmarkResult:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    device = torch.device(device)
    for _ in range(warmup_iterations):
        operation()

    synchronize(device)
    latencies_ms = []
    for _ in range(measurement_iterations):
        synchronize(device)
        started_at = time.perf_counter()
        operation()
        synchronize(device)
        latencies_ms.append((time.perf_counter() - started_at) * 1_000)

    return BenchmarkResult(
        name=name,
        device=str(device),
        dtype=str(dtype).removeprefix("torch."),
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        median_latency_ms=statistics.median(latencies_ms),
        mean_latency_ms=statistics.mean(latencies_ms),
        details=details,
    )


def write_csv(path: str | Path, results: list[BenchmarkResult]) -> None:
    if not results:
        raise ValueError("results must not be empty")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_row() for result in results]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
