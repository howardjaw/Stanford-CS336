import torch

from cs336_basics.model import softmax


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> int:
    """
    Pick the next token from one position's logits (vocab_size,).
    temperature: < 1 makes the model more confident/repetitive, > 1 more random; 0 means always take the top token.
    top_p: keep only the smallest set of most-likely tokens whose probabilities add up to at least top_p.
    """
    if temperature == 0:
        return int(logits.argmax())

    probs = softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        # Sort from most to least likely and keep tokens until the running total reaches top_p.
        sorted_probs, sorted_ids = probs.sort(descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Keep a token if the total *before* it is still below top_p (so the top token is always kept).
        keep = (cumulative - sorted_probs) < top_p
        sorted_probs = sorted_probs * keep
        choice = torch.multinomial(sorted_probs / sorted_probs.sum(), num_samples=1)
        return int(sorted_ids[choice])

    return int(torch.multinomial(probs, num_samples=1))


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
) -> list[int]:
    """
    Continue prompt_ids one token at a time: run the model, sample the next token, append it, repeat.
    Stops at eos_id (<|endoftext|>) or after max_new_tokens. Returns only the new tokens.
    """
    model.eval()
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    new_ids = []

    for _ in range(max_new_tokens):
        # The model only knows positions 0 .. context_length - 1, so feed at most the last context_length tokens.
        window = torch.tensor([ids[-context_length:]], device=device)
        logits = model(window)[0, -1]  # the prediction for the token after the last one
        logits = logits.float().cpu()  # sampling 10,000 numbers is cheap on CPU and avoids MPS quirks

        next_id = sample_next_token(logits, temperature, top_p)
        if next_id == eos_id:
            break

        ids.append(next_id)
        new_ids.append(next_id)

    return new_ids
