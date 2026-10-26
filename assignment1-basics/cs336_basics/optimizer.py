import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW (Loshchilov & Hutter, Algorithm 2), following the handout's Algorithm 1.
    Keeps two running averages per parameter:
        m: average of the gradients          (momentum: smooths noisy directions)
        v: average of the squared gradients  (sets a separate step size for each weight)
    and pulls every weight slightly toward 0 each step (decoupled weight decay).
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas: {betas}")

        # The base class stores these as each parameter group's hyperparameters.
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Per-parameter memory: the step count and both running averages.
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1  # t starts at 1
                t = state["t"]
                m, v = state["m"], state["v"]
                grad = p.grad.data

                # Line 7: bias-corrected learning rate (m and v start at 0, so early steps need a boost).
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Line 8: weight decay, decoupled from the gradient (uses lr, not lr_t).
                p.data -= lr * weight_decay * p.data

                # Lines 9-10: update the running averages in place.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)          # m = β1·m + (1 - β1)·g
                v.mul_(beta2).add_(grad * grad, alpha=1 - beta2)   # v = β2·v + (1 - β2)·g²

                # Line 11: step each weight by its own scaled amount.
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

        return loss


def get_lr_cosine_schedule(
    t: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Learning rate at step t: linear warmup from 0 to max_lr, then a cosine decay
    from max_lr to min_lr that ends at step cosine_cycle_iters, then min_lr forever.
    """
    # Warmup: rise in a straight line from 0 to max_lr.
    if t < warmup_iters:
        return t / warmup_iters * max_lr

    # Cosine decay: progress goes 0 -> 1, so 0.5 * (1 + cos(pi * progress)) goes 1 -> 0.
    if t <= cosine_cycle_iters:
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)

    # After the decay: stay at the minimum.
    return min_lr
