import pickle
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe import PAT

MAX_CACHE_SIZE = 500_000  # Maximum number of cached pre-tokens


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = list(special_tokens) if special_tokens else []

        # Create a set for efficient special-token lookup
        self.special_token_set = set(self.special_tokens)

        # Register missing special tokens in the vocabulary
        existing_tokens = set(self.vocab.values())

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in existing_tokens:
                self.vocab[max(self.vocab, default=-1) + 1] = token_bytes
                existing_tokens.add(token_bytes)

        # Reverse vocabulary: token bytes -> token ID
        self.token_to_id = {}

        for token_id, token_bytes in self.vocab.items():
            self.token_to_id[token_bytes] = token_id

        # Merge pair -> priority rank
        self.merge_ranks = {}

        for index, pair in enumerate(self.merges):
            self.merge_ranks[pair] = index

        # Compile regex for recognizing special tokens
        if self.special_tokens:
            longest_first = sorted(
                self.special_tokens,
                key=len,
                reverse=True
            )

            self.special_pattern = re.compile(
                "("
                + "|".join(re.escape(token) for token in longest_first)
                + ")"
            )
        else:
            self.special_pattern = None

        # Cache previously encoded pre-tokens
        self.cache = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        # Load vocabulary
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        # Load learned BPE merges
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        # Create and return a new Tokenizer object
        return cls(vocab, merges, special_tokens)
    
    """
    Pre-tokenize chunks.
    """
    def _split_into_pieces(self, text: str) -> Iterator[tuple[str, bool]]:
        parts = self.special_pattern.split(text) if self.special_pattern else [text]

        for part in parts:
            if part in self.special_token_set:
                yield part, True
            elif part:
                for match in re.finditer(PAT, part):
                    yield match.group(), False

    """
    Convert bytes to ids.
    If special character, find its id in self.token_to_id.
    If normal character, return _encode_pre_token method to find ids.
    """
    def _piece_to_ids(self, piece: str, is_special: bool) -> list[int]:
        if is_special:
            return[self.token_to_id[piece.encode("utf-8")]]
        return self._encode_pre_token(piece)

    """
    Encode normal pre-token to ids.
    """
    def _encode_pre_token(self, pre_token: str) -> list[int]:
        if pre_token in self.cache:
            return self.cache[pre_token]
        tokens = [bytes([value]) for value in pre_token.encode("utf-8")]

        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            best_pair = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float("inf")))

            if best_pair not in self.merge_ranks:
                break
            
            merged = best_pair[0] + best_pair[1]
            updated_tokens = []
            i = 0

            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) == best_pair:
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

    """
    A naive encode method.
    """
    def encode(self,text: str) -> list[int]:
        ids = []

        for piece, is_special in self._split_into_pieces(text):
            ids.extend(self._piece_to_ids(piece, is_special))
        
        return ids
    """
    A naive decode method.
    """
    def decode(self, ids: list[int]) ->str:
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)

        return text_bytes.decode("utf-8", errors = "replace")
    
    """
    A more memory-efficient implementation of encoding. Chunk the corpus to pieces first and 
    iterate through the pieces to encode.
    """
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""

        for chunk in iterable:
            buffer += chunk
            pieces = list(self._split_into_pieces(buffer))

            if not pieces:
                continue

            for piece, is_special in pieces[:-1]:
                yield from self._pieces_to_ids(piece, is_special)
            
            buffer = pieces[-1][0]

        for piece, is_special in self._split_into_pieces(buffer):
            yield from self._piece_to_ids(piece, is_special)




        

        


        
