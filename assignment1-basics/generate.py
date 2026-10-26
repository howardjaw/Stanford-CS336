"""
Generate text from a model trained with train.py.

Usage (from the assignment1-basics folder):
    uv run python generate.py --run-name ts_long                                  # interactive: type prompts
    uv run python generate.py --run-name ts_long --prompt "Once upon a time"      # one completion
    uv run python generate.py --run-name ts_long --temperature 0.7 --top-p 0.9

It loads ../data/runs/<run-name>/config.json (to rebuild the model) and checkpoint.pt (the trained weights).
The tokenizer is picked from the vocab size (10,000 -> TinyStories, 32,000 -> OpenWebText) unless --tokenizer is given.
"""
import argparse
import json
from pathlib import Path

import torch

from cs336_basics.decoding import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOKENIZERS = {10_000: DATA_DIR / "bpe/tinystories_10k", 32_000: DATA_DIR / "bpe/owt_32k"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="Folder under ../data/runs/ with config.json and checkpoint.pt")
    p.add_argument("--prompt", default=None, help="Text to continue; leave out to type prompts interactively")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--tokenizer", default=None, help="Folder with vocab.pkl and merges.pkl")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    run_dir = DATA_DIR / "runs" / args.run_name
    config = json.loads((run_dir / "config.json").read_text())

    tokenizer_dir = Path(args.tokenizer) if args.tokenizer else TOKENIZERS[config["vocab_size"]]
    tokenizer = Tokenizer.from_files(tokenizer_dir / "vocab.pkl", tokenizer_dir / "merges.pkl", ["<|endoftext|>"])
    eos_id = tokenizer.token_to_id[b"<|endoftext|>"]

    # Rebuild the same model the run trained, then load its weights.
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
    )
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded {args.run_name} (trained {checkpoint['iteration']:,} steps) on {device}\n")

    def complete(prompt: str) -> None:
        new_ids = generate(
            model,
            tokenizer.encode(prompt),
            max_new_tokens=args.max_tokens,
            context_length=config["context_length"],
            temperature=args.temperature,
            top_p=args.top_p,
            eos_id=eos_id,
        )
        print(prompt + tokenizer.decode(new_ids) + "\n")

    if args.prompt is not None:
        complete(args.prompt)
        return

    print("Type the start of a story and press Enter (empty line or Ctrl-C to quit).")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break
        complete(prompt)


if __name__ == "__main__":
    main()
