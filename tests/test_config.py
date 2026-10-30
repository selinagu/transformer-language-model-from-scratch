import pytest

from cs336_basics.config import (
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    TrainingConfig,
    apply_overrides,
    load_experiment_config,
    save_experiment_config,
)


def test_model_config_defaults():
    config = ModelConfig()

    assert config.vocab_size == 10_000
    assert config.d_model == 512
    assert config.model_kwargs()["num_heads"] == 16


def test_model_config_rejects_invalid_head_dimension():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=513, num_heads=16)


def test_model_config_is_immutable():
    with pytest.raises(AttributeError):
        ModelConfig().d_model = 256


def test_training_config_rejects_invalid_schedule():
    with pytest.raises(ValueError, match="warmup_iters"):
        TrainingConfig(max_iters=10, warmup_iters=11)


def test_generation_config_rejects_invalid_top_p():
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(top_p=0)


def test_loads_yaml_experiment_config(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
model:
  vocab_size: 32
  context_length: 8
  d_model: 16
  num_layers: 1
  num_heads: 4
  d_ff: 32
training:
  batch_size: 4
  max_iters: 20
  warmup_iters: 10
generation:
  temperature: 1.0
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.model.d_model == 16
    assert config.training.batch_size == 4
    assert config.generation.temperature == 1.0


def test_yaml_rejects_unknown_fields(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("model:\n  unknown_field: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown model fields"):
        load_experiment_config(config_path)


def test_explicit_overrides_leave_unspecified_values_unchanged():
    config = ExperimentConfig(training=TrainingConfig(batch_size=8, max_iters=100, warmup_iters=10))

    updated = apply_overrides(
        config,
        model_overrides={"d_model": None},
        training_overrides={"batch_size": 4, "max_iters": None},
    )

    assert updated.training.batch_size == 4
    assert updated.training.max_iters == 100
    assert updated.model == config.model


def test_saves_resolved_configuration(tmp_path):
    path = tmp_path / "resolved.yaml"
    config = ExperimentConfig(training=TrainingConfig(batch_size=4))

    save_experiment_config(config, path)

    assert load_experiment_config(path) == config


def test_model_config_rejects_invalid_attention_backend():
    with pytest.raises(ValueError, match="attention_backend"):
        ModelConfig(attention_backend="flash")


def test_model_config_rejects_odd_rope_head_dimension():
    with pytest.raises(ValueError, match="even for RoPE"):
        ModelConfig(d_model=6, num_heads=2)
