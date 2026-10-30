import argparse

import pytest

from cs336_basics.generate_from_checkpoint import resolve_model_config


MODEL_CONFIG = {
    "vocab_size": 16,
    "context_length": 8,
    "d_model": 16,
    "num_layers": 1,
    "num_heads": 4,
    "d_ff": 32,
    "theta": 10_000.0,
}


def make_args(**overrides):
    values = {field: None for field in MODEL_CONFIG}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_resolves_config_from_checkpoint_metadata():
    config = resolve_model_config(
        make_args(),
        {"metadata": {"model_config": MODEL_CONFIG}},
    )

    assert config.model_kwargs() == {**MODEL_CONFIG, "attention_backend": "naive"}


def test_resolves_legacy_config_from_cli_arguments():
    config = resolve_model_config(make_args(**MODEL_CONFIG), {})

    assert config.model_kwargs() == {**MODEL_CONFIG, "attention_backend": "naive"}


def test_legacy_checkpoint_requires_complete_cli_config():
    legacy_values = {key: value for key, value in MODEL_CONFIG.items() if key != "d_ff"}

    with pytest.raises(ValueError, match="d_ff"):
        resolve_model_config(make_args(**legacy_values), {})


def test_invalid_checkpoint_metadata_is_rejected():
    invalid = {**MODEL_CONFIG, "d_model": 15}

    with pytest.raises(ValueError, match="divisible"):
        resolve_model_config(make_args(), {"metadata": {"model_config": invalid}})
