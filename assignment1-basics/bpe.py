
"""
Implementation of BPE Tokenizer
"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):

    vocab = {} # Create an empty dictionary for vocab.

    merges = [] # Create an empty list for merge history.

    # Build the initial 256 bytestring token to integer ID map vocab list.
    for char in range(256):
        token = bytes([char])
        vocab[char] = token

    # Add special tokens to the vocab.
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[len(vocab)] = token_bytes
    
    return vocab, merges





