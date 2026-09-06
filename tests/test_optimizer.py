import math
import torch
from torch import nn

from cs336_basics.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)

def test_adamw_matches_torch():
    initial = torch.tensor([1.0, -2.0, 0.5])
    ours_parameter = nn.Parameter(initial.clone())
    torch_parameter = nn.Parameter(initial.clone())

    ours = AdamW(
        [ours_parameter],
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    reference = torch.optim.AdamW(
        [torch_parameter],
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    gradients = [
        torch.tensor([0.1, -0.2, 0.3]),
        torch.tensor([-0.4, 0.5, -0.6]),
        torch.tensor([0.2, 0.1, -0.3]),
    ]

    for gradient in gradients:
        ours_parameter.grad = gradient.clone()
        torch_parameter.grad = gradient.clone()

        ours.step()
        reference.step()

        assert torch.allclose(
            ours_parameter,
            torch_parameter,
            atol=1e-6,
            rtol=1e-5,
        )

def test_gradient_clipping_limits_total_norm():
    first = nn.Parameter(torch.zeros(2))
    second = nn.Parameter(torch.zeros(2))

    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0, 0.0])

    gradient_clipping([first, second], max_l2_norm=1.0)

    total_norm = torch.sqrt(
        (first.grad**2).sum() + (second.grad**2).sum()
    )

    assert total_norm <= 1.0 + 1e-6

def test_gradient_clipping_preserves_small_gradients():
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([0.3, 0.4])
    original = parameter.grad.clone()

    gradient_clipping([parameter], max_l2_norm=1.0)

    assert torch.equal(parameter.grad, original)

def test_cosine_schedule_boundaries():
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_iters = 2
    cosine_cycle_iters = 10

    assert math.isclose(
        get_lr_cosine_schedule(0, max_lr, min_lr, warmup_iters, cosine_cycle_iters),
        0.0,
    )
    assert math.isclose(
        get_lr_cosine_schedule(1, max_lr, min_lr, warmup_iters, cosine_cycle_iters),
        max_lr / 2,
    )
    assert math.isclose(
        get_lr_cosine_schedule(2, max_lr, min_lr, warmup_iters, cosine_cycle_iters),
        max_lr,
    )
    assert math.isclose(
        get_lr_cosine_schedule(10, max_lr, min_lr, warmup_iters, cosine_cycle_iters),
        min_lr,
    )
    assert math.isclose(
        get_lr_cosine_schedule(100, max_lr, min_lr, warmup_iters, cosine_cycle_iters),
        min_lr,
    )

def test_checkpoint_restores_model_and_optimizer(tmp_path):
    torch.manual_seed(0)

    model = nn.Linear(2, 1)
    optimizer = AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
    )

    inputs = torch.tensor([
        [1.0, 2.0],
        [-1.0, 0.5],
    ])
    targets = torch.tensor([
        [0.5],
        [-0.25],
    ])

    optimizer.zero_grad()
    loss = ((model(inputs) - targets) ** 2).mean()
    loss.backward()
    optimizer.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=7,
        out=str(checkpoint_path),
    )

    restored_model = nn.Linear(2, 1)
    restored_optimizer = AdamW(
        restored_model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    restored_iteration = load_checkpoint(
        src=str(checkpoint_path),
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert restored_iteration == 7

    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.allclose(original, restored)

    for current_model, current_optimizer in [
        (model, optimizer),
        (restored_model, restored_optimizer),
    ]:
        current_optimizer.zero_grad()
        current_loss = ((current_model(inputs) - targets) ** 2).mean()
        current_loss.backward()
        current_optimizer.step()

    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.allclose(original, restored, atol=1e-6)
