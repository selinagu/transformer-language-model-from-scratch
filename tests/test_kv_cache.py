import pytest
import torch

from cs336_basics.decoder import generate
from cs336_basics.transformer import TransformerLM


def make_model(attention_backend="naive", context_length=8):
    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=32,
        context_length=context_length,
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_ff=32,
        attention_backend=attention_backend,
    ).eval()


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_kv_cache_matches_full_forward(backend):
    model = make_model(attention_backend=backend)
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    full_logits = model(token_ids)
    first_logits, cache = model(token_ids[:, :3], use_cache=True)
    second_logits, updated_cache = model(
        token_ids[:, 3:], past_key_values=cache, use_cache=True
    )

    assert torch.allclose(first_logits, full_logits[:, :3], atol=1e-5, rtol=1e-4)
    assert torch.allclose(second_logits, full_logits[:, 3:], atol=1e-5, rtol=1e-4)
    assert len(updated_cache) == model.num_layers
    for keys, values in updated_cache:
        assert keys.shape == (1, model.num_heads, token_ids.shape[1], 4)
        assert values.shape == keys.shape


class IntegerTokenizer:
    def encode(self, prompt):
        return [1, 2, 3]

    def decode(self, token_ids):
        return " ".join(map(str, token_ids))


@pytest.mark.parametrize("context_length", [8, 4])
def test_generation_with_cache_matches_without_cache(context_length):
    model = make_model(context_length=context_length)
    tokenizer = IntegerTokenizer()

    cached = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="ignored",
        max_new_tokens=6,
        temperature=0,
        device="cpu",
        use_kv_cache=True,
    )
    uncached = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="ignored",
        max_new_tokens=6,
        temperature=0,
        device="cpu",
        use_kv_cache=False,
    )

    assert cached == uncached
