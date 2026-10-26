from collections.abc import Iterable

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Average cross-entropy loss: -log softmax(logits)[target], averaged over every position.
    logits: (..., vocab_size) raw scores. targets: (...) the correct token IDs.
    """
    # Subtract the largest logit in each row so exp() never overflows; the loss is unchanged.
    shifted = logits - logits.max(dim=-1, keepdim=True).values

    # log of the softmax denominator: log Σ exp(o[a]), one value per position: (...).
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))

    # The correct token's (shifted) logit at each position: (...).
    target_logit = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # -log softmax(o)[y] = log Σ exp(o) - o[y]   (the log and exp of the target cancel out)
    loss = log_sum_exp - target_logit

    return loss.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    If the combined L2 norm of all gradients is above max_l2_norm, scale every gradient down
    (in place) so the combined norm is just under max_l2_norm. The direction stays the same.
    """
    # Make a list first: model.parameters() is a generator and could only be looped over once.
    # Frozen parameters have no gradient, so skip them.
    grads = [p.grad for p in parameters if p.grad is not None]

    # One norm for the whole model: treat all gradients as one long vector.
    total_norm = torch.sqrt(sum((g * g).sum() for g in grads))

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
