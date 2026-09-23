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
    Pre-tokenization and frequency list.
    """
    freq = {}

    for segment in segments: # Loop over each segments separated by special tokens.
        for match in re.finditer(PAT, segment):
            piece = match.group() # Pre-tokenization.
            
            encoded = piece.encode("utf-8") # Encode pre-tokens.
            
            byte_pre_token = [] # Create a list for each encoded pre-tokens.

            for value in encoded:
                byte_pre_token.append(bytes([value])) # Loop over each byte in each pre-token.
            
            key = tuple(byte_pre_token) # Build tuple keys for each pre-tokens which are in byte representations.
            
            # Count frequency for each tuple key.
            if key in freq:
                freq[key] += 1
            else:
                freq[key] = 1
    
    """
    Count adjacent token pairs.
    """

    adj_freq = {}

    # Loop over each key tuples in freq dict.
    for sequence, count in freq.items(): 
        for i in range(len(sequence)-1): # Loop over each byte representation in every key tuple.
            
            pair = (sequence[i], sequence[i+1]) # Define adjacent pairs.

            # Count adjacent pair frequency.
            if pair in adj_freq:
                adj_freq[pair] += 1
            else:
                adj_freq[pair] = 1

    return vocab, merges





