import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save everything needed to resume training: the model's weights, the optimizer's state
    (AdamW's m, v and step counts) and the iteration number (to resume the LR schedule).
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Restore the model and optimizer from a checkpoint (in place) and return the saved iteration number.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
