import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # Regex Pattern adopted from GPT-2

"""
Implementation of BPE Tokenizer.
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
    idx = {} # A dict that records the index for each sequence.

    for segment in segments: # Loop over each segment separated by special tokens.
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

    # Build index list.        
    for sequence_id, (sequence, count) in enumerate(freq.items()):
        idx[sequence_id] = sequence, count
    

    """
    Count adjacent token pairs and build pair-to-sequence dict.
    """

    adj_freq = {}

    pair_to_sequence = {} # Build a dict that stores adjacent pair and the sequences in which they have appeared in.

    for sequence_id, (sequence, count) in idx.items():
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])

            # Count every occurrence, weighted by sequence frequency.
            if pair in adj_freq:
                adj_freq[pair] += count
            else:
                adj_freq[pair] = count

            # Record each containing sequence's ID once.
            if pair not in pair_to_sequence:
                pair_to_sequence[pair] = set()

            pair_to_sequence[pair].add(sequence_id)

    """
    Merge.
    """    
    while len(vocab) < vocab_size:
        
        best_pair = None

        best_count = -1

        # Loop over adjacent frequency dict to find the most frequently appeared pair. 
        for pair, count in adj_freq.items():
            if best_count < count:
                best_count = count
                best_pair = pair
            elif best_count == count: # Compare lexicographic order if same frequency.
                if pair > best_pair:
                    best_pair = pair
        
        # Merge.
        if best_pair is None:
            break

        merged_pair = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = merged_pair # Update the vocab dict.

        merges.append(best_pair) # Update the merges list.

        # Copy the IDs because we will modify the index sets below.
        affected_ids = pair_to_sequence[best_pair].copy()

        for sequence_id in affected_ids:
            sequence, count = idx[sequence_id]

            # 1. Remove this sequence's old pair contributions.
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])

                adj_freq[pair] -= count
                if adj_freq[pair] == 0:
                    del adj_freq[pair]

                # discard() is safe if a repeated pair already removed this ID.
                pair_to_sequence[pair].discard(sequence_id)

            # 2. Merge the winning pair within this sequence.
            updated_sequence = []
            i = 0

            while i < len(sequence):
                if (
                    i + 1 < len(sequence)
                    and (sequence[i], sequence[i + 1]) == best_pair
                ):
                    updated_sequence.append(merged_pair)
                    i += 2
                else:
                    updated_sequence.append(sequence[i])
                    i += 1

            updated_sequence = tuple(updated_sequence)

            # Preserve the sequence ID and its occurrence count.
            idx[sequence_id] = updated_sequence, count

            # 3. Add this sequence's new pair contributions.
            for i in range(len(updated_sequence) - 1):
                pair = (updated_sequence[i], updated_sequence[i + 1])

                if pair in adj_freq:
                    adj_freq[pair] += count
                else:
                    adj_freq[pair] = count

                if pair not in pair_to_sequence:
                    pair_to_sequence[pair] = set()

                pair_to_sequence[pair].add(sequence_id)
                
    return vocab, merges





