import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.config import ModelConfig, TrainingConfig
from cs336_basics.optimizer import AdamW
from cs336_basics.trainer import Trainer
from cs336_basics.transformer import TransformerLM


def make_trainer(tmp_path, *, max_iters=2):
    model_config = ModelConfig(
        vocab_size=16,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
    )
    training_config = TrainingConfig(
        batch_size=2,
        max_iters=max_iters,
        warmup_iters=0,
        eval_interval=1,
        val_batches=1,
        save_interval=1,
    )
    model = TransformerLM(**model_config.model_kwargs())
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate)
    tokens = (np.arange(64, dtype=np.uint16) % model_config.vocab_size)
    return Trainer(
        model=model,
        optimizer=optimizer,
        train_data=tokens,
        valid_data=tokens,
        model_config=model_config,
        training_config=training_config,
        device="cpu",
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_metadata={"format_version": 1},
        log_path=tmp_path / "training.csv",
    )


def test_train_step_updates_parameters(tmp_path):
    trainer = make_trainer(tmp_path)
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]

    loss = trainer.train_step()

    assert loss > 0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.model.parameters())
    )


def test_evaluate_restores_model_mode(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.model.eval()

    trainer.evaluate()

    assert not trainer.model.training


def test_fit_saves_next_step_checkpoint_and_resume_reads_it(tmp_path):
    trainer = make_trainer(tmp_path, max_iters=2)

    trainer.fit()

    assert trainer.log_path.exists()
    assert trainer.resume(trainer.checkpoint_path) == 2


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_steps(trainer, start_step: int, end_step: int) -> None:
    for step in range(start_step, end_step):
        trainer._set_learning_rate(step)
        trainer.train_step()


def test_checkpoint_restores_python_numpy_and_torch_rng_states(tmp_path):
    import random

    _seed_everything(123)
    model = torch.nn.Linear(2, 1)
    optimizer = AdamW(model.parameters())
    checkpoint_path = tmp_path / "rng.pt"
    save_checkpoint(model, optimizer, iteration=0, out=checkpoint_path)

    expected = (random.random(), np.random.random(), torch.rand(1))
    _seed_everything(999)

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = AdamW(restored_model.parameters())
    load_checkpoint(checkpoint_path, restored_model, restored_optimizer)

    actual = (random.random(), np.random.random(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_save_and_resume_matches_uninterrupted_training(tmp_path):
    total_steps = 4
    split_step = 2

    _seed_everything(2026)
    uninterrupted = make_trainer(tmp_path / "uninterrupted", max_iters=total_steps)
    _train_steps(uninterrupted, 0, total_steps)
    expected_parameters = [
        parameter.detach().clone() for parameter in uninterrupted.model.parameters()
    ]

    _seed_everything(2026)
    interrupted = make_trainer(tmp_path / "resumed", max_iters=total_steps)
    _train_steps(interrupted, 0, split_step)
    interrupted.save_checkpoint(split_step)

    restored = make_trainer(tmp_path / "resumed", max_iters=total_steps)
    assert restored.resume(interrupted.checkpoint_path) == split_step
    _train_steps(restored, split_step, total_steps)

    for expected, actual in zip(expected_parameters, restored.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
