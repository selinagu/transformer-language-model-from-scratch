import numpy as np
import torch


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= context_length:
        raise ValueError("dataset must contain more tokens than context_length")

    starts = np.random.randint(
        low=0,
        high=len(tokens) - context_length,
        size=batch_size,
    )
    inputs = np.stack([tokens[start : start + context_length] for start in starts])
    targets = np.stack([tokens[start + 1 : start + context_length + 1] for start in starts])

    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(targets).long().to(device),
    )
