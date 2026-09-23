import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # Regex Pattern adopted from GPT-2

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
    Read files and split at special tokens to segments.
    """
    with open(input_path, "r", encoding = "utf-8") as f:
        # Read file.
        text = f.read()
    
    segments = []

    # Split at special tokens.
    if not special_tokens:
        segments.append(text)
    else:
        escaped_tokens = []
        
        for token in special_tokens:
            escaped_tokens.append(re.escape(token)) # Using regex to match the literal token.
        
        pattern = "|".join(escaped_tokens) # Build a string for all special tokens with "or" for re.split().

        segments = re.split(pattern, text) # Split the text at special tokens.
    
    """
    Pretokenization.
    """
    for segment in segments:
        for match in re.finditer(PAT, segments):
            piece = match.group() # Find each and every pre-token.


    return vocab, merges





