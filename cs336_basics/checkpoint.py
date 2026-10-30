import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _capture_rng_state() -> dict[str, Any]:
    numpy_kind, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached_gaussian = (
        np.random.get_state()
    )
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_kind,
            "keys": torch.from_numpy(numpy_keys.copy()),
            "position": numpy_position,
            "has_gauss": numpy_has_gauss,
            "cached_gaussian": numpy_cached_gaussian,
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "rng_state": _capture_rng_state(),
    }
    if metadata is not None:
        checkpoint["metadata"] = metadata
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    restore_rng_state: bool = True,
) -> int:
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng_state and "rng_state" in checkpoint:
        _restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["iteration"])
