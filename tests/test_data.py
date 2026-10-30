import numpy as np
import pytest

from cs336_basics.data import get_batch


def test_get_batch_returns_shifted_windows():
    np.random.seed(0)
    tokens = np.arange(32, dtype=np.uint16)

    inputs, targets = get_batch(tokens, batch_size=4, context_length=5, device="cpu")

    assert inputs.shape == (4, 5)
    assert targets.shape == (4, 5)
    assert (targets == inputs + 1).all()


def test_get_batch_is_reproducible_with_numpy_seed():
    tokens = np.arange(32, dtype=np.uint16)

    np.random.seed(7)
    first = get_batch(tokens, batch_size=2, context_length=4, device="cpu")
    np.random.seed(7)
    second = get_batch(tokens, batch_size=2, context_length=4, device="cpu")

    assert (first[0] == second[0]).all()
    assert (first[1] == second[1]).all()


def test_get_batch_rejects_short_dataset():
    with pytest.raises(ValueError, match="more tokens"):
        get_batch(np.arange(4, dtype=np.uint16), batch_size=1, context_length=4, device="cpu")
