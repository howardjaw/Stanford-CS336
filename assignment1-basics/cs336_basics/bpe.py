import regex as re
import os
import time
from typing import BinaryIO
from multiprocessing import Pool
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # Regex Pattern adopted from GPT-2

CHUNK_SIZE_BYTES = 32 * 1024 * 1024 # Target size of each pre-tokenization chunk (32 MB).

"""
A function that helps find boundaries of chunks.
"""
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

"""
Pretokenization
"""
def pretokenize_text(text, special_tokens):
    
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
    
    # Count each pre-token string first, so each unique pre-token is only encoded once.
    piece_counts = Counter()

    for segment in segments: # Loop over each segment separated by special tokens.
        piece_counts.update(match.group() for match in re.finditer(PAT, segment)) # Pre-tokenization.

    # Build frequency list.
    freq = {}

    for piece, count in piece_counts.items():
        encoded = piece.encode("utf-8") # Encode pre-tokens.

        byte_pre_token = [] # Create a list for each encoded pre-tokens.

        for value in encoded:
            byte_pre_token.append(bytes([value])) # Loop over each byte in each pre-token.

        key = tuple(byte_pre_token) # Build tuple keys for each pre-tokens which are in byte representations.

        # Distinct strings always encode to distinct bytes, so each key appears only once here.
        freq[key] = count

    return freq

"""
Read one chunk and pre-tokenize the text.
"""
def process_chunk(input_path, start, end, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    text = chunk_bytes.decode("utf-8")
    return pretokenize_text(text, special_tokens)

def process_chunk_args(args):
    # Pool.imap_unordered passes a single argument, so unpack it here.
    return process_chunk(*args)

"""
An implementation of parallelized BPE training.
"""
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], verbose: bool = False):

    start_time = time.perf_counter()

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
    Parallel pre-tokenization.
    """
    num_processes = os.cpu_count() or 1

    # Find boundaries. Without a special token to split on, use the whole file as one chunk.
    with open(input_path, "rb") as f:
        if special_tokens:
            # Use many small chunks (about CHUNK_SIZE_BYTES each) so only a few are in memory at once.
            f.seek(0, os.SEEK_END)
            num_chunks = max(num_processes, f.tell() // CHUNK_SIZE_BYTES)
            boundaries = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode("utf-8"))
        else:
            f.seek(0, os.SEEK_END)
            boundaries = [0, f.tell()]

    chunk_args = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((input_path, start, end, special_tokens))

    # Pre-tokenize the chunks in parallel, combining each chunk's frequency list as soon as it finishes.
    freq = {}

    with Pool(max(1, min(num_processes, len(chunk_args)))) as pool:
        for chunk_freq in pool.imap_unordered(process_chunk_args, chunk_args):
            for key, count in chunk_freq.items():
                if key in freq:
                    freq[key] += count
                else:
                    freq[key] = count

    if verbose:
        print(f"Pre-tokenized {len(chunk_args)} chunks into {len(freq):,} unique pre-tokens "
              f"in {time.perf_counter() - start_time:.1f}s", flush=True)

    # Build index list.
    idx = {} # A dict that records the index for each sequence.

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

        if verbose and len(merges) % 1000 == 0:
            print(f"{len(merges):,} merges done, {time.perf_counter() - start_time:.1f}s elapsed", flush=True)

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





