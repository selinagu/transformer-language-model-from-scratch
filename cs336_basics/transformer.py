import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

KVCache = tuple[torch.Tensor, torch.Tensor]


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.normal_(self.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.weight.clamp_(-3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            self.weight.clamp_(-3.0, 3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float()
        rms = torch.sqrt(torch.mean(normalized**2, dim=-1, keepdim=True) + self.eps)
        return (normalized / rms * self.weight).to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        if not d_ff:
            d_ff = int(round(8 / 3 * d_model) / 64) * 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        return self.w2((gate * torch.sigmoid(gate)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)
        )
        positions = torch.arange(max_seq_len, device=device).float()
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        result = torch.empty_like(x)
        result[..., 0::2] = x_even * cos - x_odd * sin
        result[..., 1::2] = x_even * sin + x_odd * cos
        return result


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    stable_x = x - x_max
    return torch.exp(stable_x) / torch.sum(torch.exp(stable_x), dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(softmax(scores, dim=-1), v)


def causal_mask(
    query_length: int,
    key_length: int,
    past_length: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(past_length, past_length + query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = 10000.0,
        max_seq_len: int = 1024,
        use_rope: bool = False,
        attention_backend: str = "naive",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if attention_backend not in {"naive", "sdpa"}:
            raise ValueError("attention_backend must be 'naive' or 'sdpa'")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.attention_backend = attention_backend
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        _, sequence_length, _ = x.shape
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        if past_length + sequence_length > self.rope.cos.shape[0] if self.use_rope else False:
            raise ValueError("cached sequence exceeds max_seq_len")

        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.num_heads)

        if self.use_rope:
            positions = torch.arange(
                past_length, past_length + sequence_length, device=x.device
            )
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)

        mask = causal_mask(sequence_length, k.shape[-2], past_length, x.device)
        if self.attention_backend == "naive":
            attended = scaled_dot_product_attention(q, k, v, mask=mask)
        else:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        merged = rearrange(attended, "b h s d -> b s (h d)")
        output = self.output_proj(merged)
        if use_cache:
            return output, (k, v)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        attention_backend: str = "naive",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            use_rope=True,
            attention_backend=attention_backend,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        if use_cache:
            attention_output, key_value = self.attn(
                self.ln1(x), past_key_value=past_key_value, use_cache=True
            )
            x = x + attention_output
            return x + self.ffn(self.ln2(x)), key_value

        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        attention_backend: str = "naive",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (d_model // num_heads) % 2 != 0:
            raise ValueError("d_model / num_heads must be even for RoPE")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.attention_backend = attention_backend

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    attention_backend=attention_backend,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        past_key_values: tuple[KVCache, ...] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[KVCache, ...]]:
        if past_key_values is not None and len(past_key_values) != self.num_layers:
            raise ValueError("past_key_values must contain one entry per transformer layer")

        x = self.token_embeddings(token_ids)
        new_key_values: list[KVCache] = []
        for layer_index, layer in enumerate(self.layers):
            past_key_value = (
                None if past_key_values is None else past_key_values[layer_index]
            )
            if use_cache:
                x, key_value = layer(
                    x, past_key_value=past_key_value, use_cache=True
                )
                new_key_values.append(key_value)
            else:
                x = layer(x)

        logits = self.lm_head(self.ln_final(x))
        if use_cache:
            return logits, tuple(new_key_values)
        return logits

