import torch
import torch.nn.functional as F
from cs336_basics.transformer import softmax, RMSNorm, MultiHeadSelfAttention
from cs336_basics.optimizer import cross_entropy

def test_softmax():
    x = torch.randn(3, 5)
    actual = softmax(x, dim=-1)
    expected = torch.softmax(x, dim=-1)

    assert torch.allclose(actual, expected, atol=1e-6)

def test_cross_entropy():
    torch.manual_seed(0)

    batch_size = 2
    sequence_length = 3
    vocab_size = 5

    logits = torch.rand(batch_size, sequence_length, vocab_size)
    targets = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )
    assert torch.allclose(actual, expected, atol=1e-6)

def test_RMSNorm():
    x = torch.tensor([[3.0, 4.0]])
    norm = RMSNorm(d_model=2, eps=0.0)

    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, 0.5]))

    actual = norm(x)

    # RMS = sqrt((9 + 16) / 2) = sqrt(12.5)
    expected = torch.tensor([
        [3.0 / (12.5**0.5) * 2.0,
        4.0 / (12.5**0.5) * 0.5]
    ])

    assert torch.allclose(actual, expected, atol=1e-6)

    assert norm(x).shape == x.shape

def test_attention():
    d_model = 32
    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=4,
        use_rope=False,
    )

    x1 = torch.randn(1, 6, d_model)
    x2 = x1.clone()
    x2[:, 3:] = torch.randn_like(x2[:, 3:])

    out1 = attention(x1)
    out2 = attention(x2)

    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-6)