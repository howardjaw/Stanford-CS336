import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

BATCH_SIZE = 1_000_000  # how many IDs to collect before writing them to disk


def encode_file(tokenizer, input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch = []
    total = 0
    start = time.perf_counter()

    with open(input_path, encoding="utf-8") as f_in, open(output_path, "wb") as f_out:
        for token_id in tokenizer.encode_iterable(f_in):   # reads the file line by line
            batch.append(token_id)

            if len(batch) == BATCH_SIZE:
                np.array(batch, dtype=np.uint16).tofile(f_out)  # write 1M IDs as 2-byte numbers
                total += len(batch)
                if total % 100_000_000 == 0:
                    print(f"  {total:,} tokens so far, {time.perf_counter() - start:.0f}s", flush=True)
                batch = []

        np.array(batch, dtype=np.uint16).tofile(f_out)  # write whatever is left
        total += len(batch)

    print(f"{input_path.name}: {total:,} tokens in {time.perf_counter() - start:.0f}s")

DATA = Path("../data")
ts_tok = Tokenizer.from_files(DATA / "bpe/tinystories_10k/vocab.pkl", DATA / "bpe/tinystories_10k/merges.pkl", ["<|endoftext|>"])
owt_tok = Tokenizer.from_files("../data/bpe/owt_32k/vocab.pkl",         "../data/bpe/owt_32k/merges.pkl",         ["<|endoftext|>"])
# encode_file(ts_tok, DATA / "TinyStoriesV2-GPT4-valid.txt", DATA / "tokens/tinystories_valid.bin")

# ids = np.memmap(DATA / "tokens/tinystories_valid.bin", dtype=np.uint16, mode="r")
# eot = ts_tok.token_to_id[b"<|endoftext|>"]

# print(len(ids))                       # ~5.6M
# print(ids.max())                      # must be < 10,000
# print((ids == eot).sum())             # must be 27,630, the number of <|endoftext|> in the file
# print(ts_tok.decode(ids[:60].tolist())) # should match the start of the text file

encode_file(owt_tok, DATA / "owt_valid.txt",                DATA / "tokens/owt_valid.bin")
encode_file(ts_tok,  DATA / "TinyStoriesV2-GPT4-train.txt", DATA / "tokens/tinystories_train.bin")
encode_file(owt_tok, DATA / "owt_train.txt",                DATA / "tokens/owt_train.bin")
