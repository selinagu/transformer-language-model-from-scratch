import torch

from cs336_basics.transformer import softmax


def top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    keep = torch.ones_like(cumulative_probs, dtype=torch.bool)
    keep[1:] = cumulative_probs[:-1] <= top_p

    filtered_probs = torch.zeros_like(probs)
    filtered_probs[sorted_indices[keep]] = probs[sorted_indices[keep]]
    return filtered_probs / filtered_probs.sum()


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    device="cuda:0",
    use_kv_cache: bool = True,
):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    model.eval()

    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    past_key_values = None

    for _ in range(max_new_tokens):
        if not use_kv_cache:
            logits = model(ids[:, -model.context_length:])
        elif past_key_values is None or past_key_values[0][0].shape[-2] >= model.context_length:
            input_ids = ids[:, -model.context_length:]
            logits, past_key_values = model(input_ids, use_cache=True)
        else:
            logits, past_key_values = model(
                ids[:, -1:], past_key_values=past_key_values, use_cache=True
            )

        next_logits = logits[0, -1, :]
        if temperature == 0:
            next_id = torch.argmax(next_logits)
        else:
            probs = softmax(next_logits / temperature, dim=-1)
            if top_p is not None:
                probs = top_p_filtering(probs, top_p)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(0)

        ids = torch.cat((ids, next_id.view(1, 1)), dim=1)
        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return tokenizer.decode(ids.squeeze(0).tolist())
