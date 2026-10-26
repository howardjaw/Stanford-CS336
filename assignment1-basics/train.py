"""
Train a Transformer language model on a tokenized dataset (a .bin file of uint16 token IDs).

Defaults follow the handout's TinyStories setup and its Apple Silicon tip:
batch 32 x 5,000 steps x 256 context = ~41M tokens, with the cosine schedule ending at the last step.

Usage (from the assignment1-basics folder):
    uv run python train.py --run-name ts_baseline
    uv run python train.py --run-name ts_lr3e-3 --lr-max 3e-3
    uv run python train.py --run-name ts_baseline --resume          # continue from its last checkpoint

Each run writes to ../data/runs/<run-name>/:
    config.json    the settings used (needed to rebuild the model for generation)
    log.jsonl      one line per log/eval: step, train/val loss, learning rate, elapsed seconds
    checkpoint.pt  model + optimizer + step, saved every --checkpoint-every steps and at the end
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Run and data
    p.add_argument("--run-name", required=True, help="Output folder name under ../data/runs/")
    p.add_argument("--train-data", default=str(DATA_DIR / "tokens/tinystories_train.bin"))
    p.add_argument("--valid-data", default=str(DATA_DIR / "tokens/tinystories_valid.bin"))
    p.add_argument("--resume", action="store_true", help="Continue from the run's checkpoint.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (aot_eager on mps)")

    # Model (handout §7.2.1 TinyStories settings)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10_000.0)

    # Optimizer and schedule
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=5_000)
    p.add_argument("--lr-max", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Logging and checkpointing
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=1_000)
    return p.parse_args()


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate(model: torch.nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Average validation loss over a fixed set of batches (the same ones every time, so curves are comparable)."""
    model.eval()
    total = sum(cross_entropy(model(x), y).item() for x, y in batches)
    model.train()
    return total / len(batches)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = DATA_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Memory-mapped token files: only the windows we sample are read from disk.
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    valid_data = np.memmap(args.valid_data, dtype=np.uint16, mode="r")

    # A fixed set of validation batches, sampled once.
    val_batches = [get_batch(valid_data, args.batch_size, args.context_length, device) for _ in range(args.eval_batches)]

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"Resumed from {checkpoint_path} at step {start_step}")

    # Compiling wraps the model; checkpoints still save the plain `model` so its weight names stay clean.
    train_model = model
    if args.compile:
        train_model = torch.compile(model, backend="aot_eager" if device == "mps" else "inductor")

    num_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch_size * args.context_length
    print(f"device {device} | {num_params:,} parameters | {tokens_per_step:,} tokens/step | "
          f"{args.max_steps:,} steps = {tokens_per_step * args.max_steps:,} tokens")

    log_file = open(run_dir / "log.jsonl", "a")
    start_time = time.perf_counter()

    for step in range(start_step, args.max_steps):
        # 1. Learning rate for this step; the cosine decay ends exactly at the last step.
        lr = get_lr_cosine_schedule(step, args.lr_max, args.lr_min, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. Forward: a random batch, the model's predictions, and the loss.
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        loss = cross_entropy(train_model(x), y)

        # 3. Backward, clip, update.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # 4. Logging, evaluation and checkpoints.
        done = step + 1
        is_last = done == args.max_steps
        record = None

        if done % args.log_every == 0 or is_last:
            train_loss = loss.item()
            if not math.isfinite(train_loss):
                raise RuntimeError(f"Loss became {train_loss} at step {done}; try a smaller --lr-max.")
            record = {"step": done, "train_loss": train_loss, "lr": lr}

        if done % args.eval_every == 0 or is_last:
            record = record or {"step": done, "lr": lr}
            record["val_loss"] = evaluate(model, val_batches)

        if record is not None:
            elapsed = time.perf_counter() - start_time
            record["elapsed_s"] = round(elapsed, 1)
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            message = f"step {done:6,}/{args.max_steps:,} | lr {lr:.2e} | {elapsed / 60:5.1f} min"
            if "train_loss" in record:
                message += f" | train {record['train_loss']:.3f}"
            if "val_loss" in record:
                message += f" | val {record['val_loss']:.3f} (ppl {math.exp(record['val_loss']):.1f})"
            print(message, flush=True)

        if done % args.checkpoint_every == 0 or is_last:
            save_checkpoint(model, optimizer, done, checkpoint_path)

    log_file.close()
    print(f"Done. Checkpoint, config and log are in {run_dir}")


if __name__ == "__main__":
    main()
