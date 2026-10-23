"""
Train a byte-level BPE tokenizer on TinyStories or OpenWebText, save the vocab and merges,
and report time, memory and the longest tokens.

Usage (from the assignment1-basics folder; `-m` puts this folder on the import path):
    uv run python -m scripts.train_bpe tinystories
    uv run python -m scripts.train_bpe owt
    uv run python -m scripts.train_bpe tinystories --profile   # also print where the time went
"""
import argparse
import cProfile
import pickle
import pstats
import resource
import sys
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Dataset name -> (training file, vocab size from the handout).
DATASETS = {
    "tinystories": (DATA_DIR / "TinyStoriesV2-GPT4-train.txt", 10_000),
    "owt": (DATA_DIR / "owt_train.txt", 32_000),
}

SPECIAL_TOKENS = ["<|endoftext|>"]


def peak_memory_gb(who):
    # ru_maxrss is in bytes on macOS but in kilobytes on Linux.
    peak = resource.getrusage(who).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return peak / 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--profile", action="store_true", help="Print the functions that took the most time.")
    args = parser.parse_args()

    input_path, vocab_size = DATASETS[args.dataset]
    out_dir = DATA_DIR / "bpe" / f"{args.dataset}_{vocab_size // 1000}k"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on {input_path.name} with vocab size {vocab_size:,}", flush=True)

    profiler = cProfile.Profile() if args.profile else None
    start = time.perf_counter()
    if profiler:
        profiler.enable()

    vocab, merges = train_bpe(str(input_path), vocab_size, SPECIAL_TOKENS, verbose=True)

    if profiler:
        profiler.disable()
    elapsed = time.perf_counter() - start

    with open(out_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(out_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    print()
    print(f"Training time: {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    print(f"Peak memory: main process {peak_memory_gb(resource.RUSAGE_SELF):.2f} GB, "
          f"largest worker process {peak_memory_gb(resource.RUSAGE_CHILDREN):.2f} GB")
    print(f"Vocab size: {len(vocab):,}, merges: {len(merges):,}")
    print(f"Saved to {out_dir}")

    print("\nLongest tokens:")
    for token in sorted(vocab.values(), key=len, reverse=True)[:5]:
        print(f"  {len(token):3d} bytes  {token!r}")

    if profiler:
        print("\nTop functions by time spent inside them (worker processes show up as waiting):")
        pstats.Stats(profiler).sort_stats("tottime").print_stats(12)


if __name__ == "__main__":
    main()
