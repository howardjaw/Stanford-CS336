import pickle
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe import PAT

MAX_CACHE_SIZE = 500_000 # Most pre-tokens we remember the encoding of (keeps memory bounded).


class Tokenizer:
    """
    A byte-level BPE tokenizer. It uses the vocab and merges learned by train_bpe to turn
    text into token IDs (encode) and token IDs back into text (decode).
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # Copy the vocab so adding special tokens below doesn't change the caller's dict.
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = list(special_tokens) if special_tokens else []
        self.special_token_set = set(self.special_tokens)

        # Add any special token that isn't in the vocab yet, using the next unused ID.
        existing_tokens = set(self.vocab.values())

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in existing_tokens:
                self.vocab[max(self.vocab, default=-1) + 1] = token_bytes
                existing_tokens.add(token_bytes)

        # Reverse lookup, token bytes -> ID. encode needs it to turn merged bytes into IDs.
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        # Each merge's position in the list, e.g. {(b't', b'h'): 0, ...}.
        # A lower rank means the merge was learned earlier, so encode applies it first.
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        # Regex that finds special tokens in text.
        # Longest first, so "<|endoftext|><|endoftext|>" is matched before "<|endoftext|>".
        # The outer parentheses make re.split keep the special tokens instead of dropping them.
        if self.special_tokens:
            longest_first = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile("(" + "|".join(re.escape(token) for token in longest_first) + ")")
        else:
            self.special_pattern = None

        # Pre-token string -> its token IDs. Common words repeat a lot, so this saves re-merging them.
        self.cache = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        # Load the vocab and merges saved by scripts/train_bpe.py (pickle files), then build a Tokenizer.
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids = []

        for piece, is_special in self._split_into_pieces(text):
            ids.extend(self._piece_to_ids(piece, is_special))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Encode a stream of text (e.g. a file, one line at a time) without loading it all into memory.
        # The last piece of each chunk is held back, because it might continue in the next chunk:
        # for example "a \n" + "b" must split as "a", " ", "\n", "b", not "a", " \n", "b".
        # Assumes a special token is never cut in half between two chunks, which holds for file lines.
        buffer = ""

        for chunk in iterable:
            buffer += chunk
            pieces = list(self._split_into_pieces(buffer))

            if not pieces:
                continue

            for piece, is_special in pieces[:-1]:
                yield from self._piece_to_ids(piece, is_special)

            buffer = pieces[-1][0]

        # Nothing more is coming, so the held-back piece is final.
        for piece, is_special in self._split_into_pieces(buffer):
            yield from self._piece_to_ids(piece, is_special)

    def decode(self, ids: list[int]) -> str:
        # Join every token's bytes, then turn the bytes into text.
        # One token can be half of a multi-byte character, so invalid bytes become U+FFFD instead of crashing.
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)

        return text_bytes.decode("utf-8", errors="replace")

    def _split_into_pieces(self, text: str) -> Iterator[tuple[str, bool]]:
        # Yield (piece, is_special) in order: each special token whole, everything else as GPT-2 pre-tokens.
        parts = self.special_pattern.split(text) if self.special_pattern else [text]

        for part in parts:
            if part in self.special_token_set:
                yield part, True
            elif part: # re.split can produce empty strings between back-to-back special tokens.
                for match in re.finditer(PAT, part):
                    yield match.group(), False

    def _piece_to_ids(self, piece: str, is_special: bool) -> list[int]:
        if is_special:
            return [self.token_to_id[piece.encode("utf-8")]]

        return self._encode_pre_token(piece)

    def _encode_pre_token(self, pre_token: str) -> list[int]:
        if pre_token in self.cache:
            return self.cache[pre_token]

        # Start from single bytes, exactly like training.
        tokens = [bytes([value]) for value in pre_token.encode("utf-8")]

        # Repeatedly merge the adjacent pair that was learned earliest, until no learned pair is left.
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_pair = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float("inf")))

            if best_pair not in self.merge_ranks:
                break # None of the pairs was ever learned, so we're done.

            # Merge every occurrence of the best pair, left to right.
            merged = best_pair[0] + best_pair[1]
            updated_tokens = []
            i = 0

            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    updated_tokens.append(merged)
                    i += 2
                else:
                    updated_tokens.append(tokens[i])
                    i += 1

            tokens = updated_tokens

        ids = [self.token_to_id[token] for token in tokens]

        if len(self.cache) < MAX_CACHE_SIZE:
            self.cache[pre_token] = ids

        return ids
