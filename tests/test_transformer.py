import pytest
import torch
import torch.nn.functional as F

from cs336_basics.optimizer import cross_entropy
from cs336_basics.transformer import (
    MultiHeadSelfAttention,
    RMSNorm,
    TransformerLM,
    softmax,
)


def test_softmax():
    x = torch.randn(3, 5)
    actual = softmax(x, dim=-1)
    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_cross_entropy():
    torch.manual_seed(0)
    batch_size, sequence_length, vocab_size = 2, 3, 5
    logits = torch.rand(batch_size, sequence_length, vocab_size)
    targets = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_rms_norm():
    x = torch.tensor([[3.0, 4.0]])
    norm = RMSNorm(d_model=2, eps=0.0)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, 0.5]))

    expected = torch.tensor([[3.0 / (12.5**0.5) * 2.0, 4.0 / (12.5**0.5) * 0.5]])
    assert torch.allclose(norm(x), expected, atol=1e-6)
    assert norm(x).shape == x.shape


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_attention_is_causal(backend):
    d_model = 32
    attention = MultiHeadSelfAttention(
        d_model=d_model, num_heads=4, use_rope=False, attention_backend=backend
    )
    x1 = torch.randn(1, 6, d_model)
    x2 = x1.clone()
    x2[:, 3:] = torch.randn_like(x2[:, 3:])

    assert torch.allclose(attention(x1)[:, :3], attention(x2)[:, :3], atol=1e-6)


def test_sdpa_matches_naive_logits_and_gradients():
    torch.manual_seed(0)
    model_kwargs = {
        "vocab_size": 32, "context_length": 8, "d_model": 16,
        "num_layers": 2, "num_heads": 4, "d_ff": 32,
    }
    naive = TransformerLM(**model_kwargs, attention_backend="naive")
    sdpa = TransformerLM(**model_kwargs, attention_backend="sdpa")
    sdpa.load_state_dict(naive.state_dict())

    token_ids = torch.randint(0, model_kwargs["vocab_size"], (2, 6))
    naive_logits = naive(token_ids)
    sdpa_logits = sdpa(token_ids)
    assert torch.allclose(naive_logits, sdpa_logits, atol=1e-5, rtol=1e-4)

    naive_logits.square().mean().backward()
    sdpa_logits.square().mean().backward()
    for naive_parameter, sdpa_parameter in zip(naive.parameters(), sdpa.parameters()):
        assert torch.allclose(naive_parameter.grad, sdpa_parameter.grad, atol=1e-5, rtol=1e-4)


def test_transformer_rejects_odd_rope_head_dimension():
    with pytest.raises(ValueError, match="even for RoPE"):
        TransformerLM(
            vocab_size=8,
            context_length=4,
            d_model=6,
            num_layers=1,
            num_heads=2,
            d_ff=8,
        )
