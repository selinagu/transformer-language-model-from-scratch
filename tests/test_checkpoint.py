import torch
from torch import nn

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.optimizer import AdamW


def test_checkpoint_preserves_optional_metadata(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    metadata = {"format_version": 1, "model_config": {"d_model": 16}}

    save_checkpoint(model, optimizer, iteration=3, out=path, metadata=metadata)

    payload = torch.load(path, weights_only=True)
    assert payload["metadata"] == metadata

    restored_model = nn.Linear(2, 1)
    restored_optimizer = AdamW(restored_model.parameters())
    assert load_checkpoint(path, restored_model, restored_optimizer) == 3


def test_checkpoint_omits_metadata_when_not_provided(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = AdamW(model.parameters())
    path = tmp_path / "legacy.pt"

    save_checkpoint(model, optimizer, iteration=1, out=path)

    assert "metadata" not in torch.load(path, weights_only=True)
