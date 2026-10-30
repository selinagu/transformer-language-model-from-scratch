import pytest
import torch

from cs336_basics.decoder import generate, top_p_filtering


class IntegerTokenizer:
    def encode(self, prompt: str) -> list[int]:
        return [0]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(map(str, token_ids))


class EosModel(torch.nn.Module):
    context_length = 8

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], 3),
            float("-inf"),
            device=token_ids.device,
        )
        logits[..., 2] = 0
        return logits


def test_top_p_keeps_token_that_reaches_probability_threshold():
    filtered = top_p_filtering(torch.tensor([0.5, 0.3, 0.2]), top_p=0.7)

    torch.testing.assert_close(filtered, torch.tensor([0.625, 0.375, 0.0]))


def test_generate_stops_immediately_after_eos():
    model = EosModel()

    text = generate(
        model,
        IntegerTokenizer(),
        "prompt",
        max_new_tokens=5,
        eos_token_id=2,
        temperature=0,
        device="cpu",
        use_kv_cache=False,
    )

    assert text == "0 2"
    assert model.calls == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
    ],
)
def test_generate_rejects_invalid_sampling_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate(
            EosModel(),
            IntegerTokenizer(),
            "prompt",
            device="cpu",
            use_kv_cache=False,
            **({"max_new_tokens": 1} | kwargs),
        )
