import regex as re
from collections.abc import Iterable, Iterator
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] =None
    ):
        self.vocab = vocab
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merges_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        if special_tokens is not None:
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.bytes_to_id:
                    new_id = max(self.vocab.keys()) + 1
                    self.vocab[new_id] = token_bytes
                    self.bytes_to_id[token_bytes] = new_id


    @staticmethod
    def merge_pair_in_tokens(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        """
        Merge pair in a token.
        """
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return tuple(new_tokens)


    def encode_pretoken(self, tokens: tuple[bytes, ...]) -> list[int]:
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            candidate_pairs = [pair for pair in pairs if pair in self.merges_ranks] 

            if not candidate_pairs:
                break

            best_pair = min(candidate_pairs, key = lambda pair: self.merges_ranks[pair])
            tokens = self.merge_pair_in_tokens(tokens, best_pair)

        return [self.bytes_to_id[token] for token in tokens]





    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Load the trained tokenizer.
        """
        import pickle

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)




    def encode(self, text: str) -> list[int]:
        """
        Encode text.
        """
        
        # Split on special tokens
        special_tokens_sorted = sorted(self.special_tokens, key = len, reverse = True)
        
        if not special_tokens_sorted:
            parts = [text]
        
        else:
            pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens_sorted) + ")"

            parts = re.split(pattern, text)

        # Pretokenize + naive encoding
        encoded = []
        for part in parts:
            if part in self.special_tokens:
                special_token_bytes = part.encode("utf-8")
                encoded.append(self.bytes_to_id[special_token_bytes])
                continue

            for match in re.finditer(PAT, part):
                pretoken = match.group()
                token_bytes = pretoken.encode("utf-8")
                tokens = tuple(bytes([b]) for b in token_bytes)
                encoded.extend(self.encode_pretoken(tokens))

        return encoded




    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode iterable strings.
        """
        buffer = ""

        special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        max_special_len = max((len(tok) for tok in self.special_tokens), default = 0)

        for chunk in iterable:
            buffer += chunk

            while buffer:
                old_len = len(buffer)

                # 1. If buffer starts with a special token, emit it immediately
                matched_special = None
                for tok in special_tokens_sorted:
                    if buffer.startswith(tok):
                        matched_special = tok
                        break

                if matched_special is not None:
                    yield self.bytes_to_id[matched_special.encode("utf-8")]
                    buffer = buffer[len(matched_special):]
                    continue


                # 2. If a complete special token appears later in buffer,
                # encode the text before it, then keep the special token
                first_special_pos = None

                for tok in special_tokens_sorted:
                    pos = buffer.find(tok)
                    if pos != -1:
                        if first_special_pos is None or pos < first_special_pos:
                            first_special_pos = pos

                if first_special_pos is not None:
                    if first_special_pos > 0:
                        prefix = buffer[:first_special_pos]
                        buffer = buffer[first_special_pos:]
                        yield from self.encode(prefix)
                        continue

                    
                # 3. No complete special token in buffer.
                # Keep enough suffix so a special token split across chunks is not broken
                if max_special_len > 0:
                    cutoff = max(0, len(buffer) - max_special_len + 1)
                else:
                    cutoff = len(buffer)
                    
                matches = list(re.finditer(PAT, buffer))
                candidate_boundaries = [
                    m.start() for m in matches if m.start() < cutoff
                ]

                if not candidate_boundaries:
                    break

                safe_end = candidate_boundaries[-1]

                if safe_end > 0:
                    safe_text = buffer[:safe_end]
                    buffer = buffer[safe_end:]
                    yield from self.encode(safe_text)

                if len(buffer) >= old_len:
                    break

        if buffer:
            yield from self.encode(buffer)


    def decode(self, ids: list[int]) -> str:
        """
        Decoding.
        """
        text_bytes = b"".join(self.vocab[id] for id in ids)
        text = text_bytes.decode("utf-8", errors = "replace")

        return text