import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample batch_size random windows of context_length tokens from one long token array.
    Returns (inputs, targets), both (batch_size, context_length); targets are the inputs shifted by one.
    Works with np.memmap, so only the sampled windows are read from disk.
    """
    # Valid starts are 0 .. len - context_length - 1, so the target window still fits.
    # (randint's upper bound is excluded.)
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    # A grid of positions: row i is [start_i, start_i + 1, ..., start_i + context_length - 1].
    positions = starts[:, None] + np.arange(context_length)

    inputs = dataset[positions]
    targets = dataset[positions + 1]  # the next token for every position

    # Token files are uint16; embedding lookups need int64 ("long") IDs.
    inputs = torch.from_numpy(inputs.astype(np.int64)).to(device)
    targets = torch.from_numpy(targets.astype(np.int64)).to(device)
    return inputs, targets
