import random
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer

ts_text = Path("../data/TinyStoriesV2-GPT4-valid.txt").read_text(encoding="utf-8")
owt_text = Path("../data/owt_valid.txt").read_text(encoding="utf-8")


ts_tok = Tokenizer.from_files("../data/bpe/tinystories_10k/vocab.pkl","../data/bpe/tinystories_10k/merges.pkl",["<|endoftext|>"])
owt_tok = Tokenizer.from_files("../data/bpe/owt_32k/vocab.pkl",         "../data/bpe/owt_32k/merges.pkl",         ["<|endoftext|>"])

ts_docs = [doc for doc in ts_text.split("<|endoftext|>") if doc.strip()]
owt_docs = [doc for doc in owt_text.split("<|endoftext|>") if doc.strip()]

ts_sample = random.Random(42).sample(ts_docs, 10)
owt_sample = random.Random(42).sample(owt_docs, 10)

def compression_ratio(tokenizer, documents):
    total_bytes = 0
    total_tokens = 0

    for doc in documents:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    
    return total_bytes / total_tokens

ts_ratio = compression_ratio(ts_tok, ts_sample)
owt_ratio = compression_ratio(owt_tok, owt_sample)

print(f"TinyStories: {ts_ratio:.3f} bytes/token")
print(f"OpenWebText: {owt_ratio:.3f} bytes/token")



