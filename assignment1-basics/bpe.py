import regex as re

"""
Implementation of BPE Tokenizer
"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):

    """
    Initializing vocab.
    """
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

    """
    Reading files and split at special tokens.
    """
    with open(input_path, "r", encoding = "utf-8") as f:
        # Read file.
        text = f.read()
    
    chunked = []

    # Split at special tokens.
    if not special_tokens:
        chunked.append(text)
    else:
        escaped_tokens = []
        
        for token in special_tokens:
            escaped_tokens.append(re.escape(token)) # Using regex to match the literal token.
        
        pattern = "|".join(escaped_tokens) # Build a string for all special tokens with "or" for re.split().

        chunked = re.split(pattern, text) # Split the text at special tokens
    

    return vocab, merges





