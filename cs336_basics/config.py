from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    theta: float = 10_000.0
    attention_backend: str = "naive"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("d_model / num_heads must be even for RoPE")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.attention_backend not in {"naive", "sdpa"}:
            raise ValueError("attention_backend must be 'naive' or 'sdpa'")

    def model_kwargs(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    max_iters: int = 5000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 500
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0
    eval_interval: int = 100
    val_batches: int = 20
    save_interval: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must be non-negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if not 0 <= self.warmup_iters <= self.max_iters:
            raise ValueError("warmup_iters must be between 0 and max_iters")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive when provided")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.val_batches <= 0:
            raise ValueError("val_batches must be positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section_values(
    raw_config: Mapping[str, Any],
    section: str,
    config_type: type[ModelConfig] | type[TrainingConfig] | type[GenerationConfig],
) -> dict[str, Any]:
    values = raw_config.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"{section} must be a mapping")

    unknown = set(values) - set(config_type.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {section} fields: {sorted(unknown)}")
    return values


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("configuration root must be a mapping")

    unknown_sections = set(raw_config) - {"model", "training", "generation"}
    if unknown_sections:
        raise ValueError(f"unknown configuration sections: {sorted(unknown_sections)}")

    return ExperimentConfig(
        model=ModelConfig(**_section_values(raw_config, "model", ModelConfig)),
        training=TrainingConfig(**_section_values(raw_config, "training", TrainingConfig)),
        generation=GenerationConfig(**_section_values(raw_config, "generation", GenerationConfig)),
    )


def apply_overrides(
    config: ExperimentConfig,
    *,
    model_overrides: Mapping[str, Any] | None = None,
    training_overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    model_values = {key: value for key, value in (model_overrides or {}).items() if value is not None}
    training_values = {
        key: value for key, value in (training_overrides or {}).items() if value is not None
    }
    return replace(
        config,
        model=replace(config.model, **model_values),
        training=replace(config.training, **training_values),
    )


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
