import os
import regex as re
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO
from collections import defaultdict
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """
    Initialize the vocabulary.
    """
    vocab = {i: bytes([i]) for i in range(256)}

    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    return vocab

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split the text on special tokens.
    """
    if not special_tokens:
        return [text]
    
    pattern = "|".join(re.escape(tok) for tok in special_tokens)

    return re.split(pattern, text)



def pretokenize(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """
    Pretokenize using a regex-based pre-tokenizer.
    """
    pretoken_counts = Counter()
    
    for part in split_on_special_tokens(text, special_tokens):
        for match in re.finditer(PAT, part):
            pretoken = match.group()
            token_bytes = pretoken.encode("utf-8")
            byte_tuple = tuple(bytes([b]) for b in token_bytes)
            pretoken_counts[byte_tuple] += 1

    return pretoken_counts



def compute_pair_counts(
        pretoken_counts: dict[tuple[bytes, ...], int]
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set]]:
    """
    Get pair counts and pair-to-tokens after pretokenization.
    """
    pair_counts = Counter()
    pair_to_tokens = defaultdict(set)

    for tokens, count in pretoken_counts.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count
            pair_to_tokens[pair].add(tokens)

    return pair_counts, pair_to_tokens



class ReversePair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

# def make_heap_entry(pair: tuple[bytes, bytes], freq: int):
#     return (-freq, ReversePair(pair), pair)

# def build_pair_heap(pair_counts: Counter[tuple[bytes, bytes]]):
#     heap = []
#     for pair, freq in pair_counts.items():
#         if freq > 0:
#             heapq.heappush(heap, make_heap_entry(pair, freq))

#     return heap

# def get_best_pair_from_heap(heap, pair_counts):
#     """
#     Use heap to get the most frequent pair.
#     """
#     while heap:
#         neg_freq, _, pair = heapq.heappop(heap)
#         freq = -neg_freq

#         if pair in pair_counts and pair_counts[pair] == freq:
#             return pair

#     return None


def get_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes] | None:
    """
    Naive way to get the most frequent pair.
    """
    if not pair_counts:
        return None 
    return max(pair_counts.items(), key = lambda x: (x[1], x[0]))[0]



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



def apply_merge(
        pretoken_counts: dict[tuple[bytes, ...], int], 
        pair: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
    """
    Apply naive merge on pretokens.
    """
    new_counts = Counter()

    for tokens, count in pretoken_counts.items():
        new_tokens = merge_pair_in_tokens(tokens, pair)
        new_counts[new_tokens] += count

    return new_counts



# def apply_merge(
#         pretoken_counts: dict[tuple[bytes, ...], int], 
#         pair_counts: Counter[tuple[bytes, bytes]],
#         pair_to_tokens: dict[tuple[bytes, bytes], set],
#         best_pair: tuple[bytes, bytes],
#         heap
# ) -> tuple[dict[tuple[bytes, ...], int],
#     Counter[tuple[bytes, bytes]],
#     dict[tuple[bytes, bytes], set]]:
#     """
#     Apply optimized merge on pretokens.
#     """
#     affected_tokens = list(pair_to_tokens[best_pair])

#     for tok in affected_tokens:
#         if tok not in pretoken_counts:
#             continue
#         freq = pretoken_counts[tok]

#         # Remove pair counts of the old token in pair_counts and pair_to_tokens
#         for i in range(len(tok) - 1):
#             old_pair = (tok[i], tok[i + 1])

#             pair_counts[old_pair] -= freq
#             pair_to_tokens[old_pair].discard(tok)

#             if pair_counts[old_pair] <= 0:
#                 del pair_counts[old_pair]


#             # Push new entry of old_pair into heap
#             else:
#                 heapq.heappush(heap, make_heap_entry(old_pair, pair_counts[old_pair]))

#         del pretoken_counts[tok]

#         # Generate the new merged token
#         new_tok = merge_pair_in_tokens(tok, best_pair)

#         pretoken_counts[new_tok] = pretoken_counts.get(new_tok, 0) + freq
    
#         # Add pair counts of the new token
#         for i in range(len(new_tok) - 1):
#             new_pair = (new_tok[i], new_tok[i + 1])
#             pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq
#             pair_to_tokens[new_pair].add(new_tok)

#             # Push the new entry of new_pair into heap
#             if pair_counts[new_pair] > 0:
#                 heapq.heappush(heap, make_heap_entry(new_pair, pair_counts[new_pair]))

#     return pretoken_counts, pair_counts, pair_to_tokens



def apply_merge(
        pretoken_counts: dict[tuple[bytes, ...], int], 
        pair_counts: Counter[tuple[bytes, bytes]],
        pair_to_tokens: dict[tuple[bytes, bytes], set],
        best_pair: tuple[bytes, bytes],
        
) -> tuple[dict[tuple[bytes, ...], int],
    Counter[tuple[bytes, bytes]],
    dict[tuple[bytes, bytes], set]]:
    """
    Apply optimized merge on pretokens.
    """
    affected_tokens = list(pair_to_tokens[best_pair])

    for tok in affected_tokens:
        if tok not in pretoken_counts:
            continue
        freq = pretoken_counts[tok]

        # Remove pair counts of the old token in pair_counts and pair_to_tokens
        for i in range(len(tok) - 1):
            old_pair = (tok[i], tok[i + 1])

            pair_counts[old_pair] -= freq
            pair_to_tokens[old_pair].discard(tok)

            if pair_counts[old_pair] <= 0:
                del pair_counts[old_pair]


        del pretoken_counts[tok]

        # Generate the new merged token
        new_tok = merge_pair_in_tokens(tok, best_pair)

        pretoken_counts[new_tok] = pretoken_counts.get(new_tok, 0) + freq
    
        # Add pair counts of the new token
        for i in range(len(new_tok) - 1):
            new_pair = (new_tok[i], new_tok[i + 1])
            pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq
            pair_to_tokens[new_pair].add(new_tok)

    return pretoken_counts, pair_counts, pair_to_tokens

        



def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer.
    Returns vocab, the tokenizer vocabulary; merges, a list of BPE merges produced from training.
    """
    vocab = init_vocab(special_tokens)
    merges = []

    pretoken_counts = pretokenize_multiprocessing(
        input_path,
        special_tokens,
        num_processes=4
    )

    next_id = len(vocab)

    pair_counts, pair_to_tokens = compute_pair_counts(pretoken_counts)


    while len(vocab) < vocab_size:

        best_pair = get_best_pair(pair_counts)
        if best_pair is None:
            break

        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1

        merges.append(best_pair)

        pretoken_counts, pair_counts, pair_to_tokens = apply_merge(
            pretoken_counts, 
            pair_counts, 
            pair_to_tokens, 
            best_pair,
            
        )

    return vocab, merges



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



def count_pretokens_in_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """
    Run pre-tokenization on the chunk and store the counts for each pre-token.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

    return pretokenize(chunk, special_tokens)
    


def pretokenize_multiprocessing(
        input_path: str, 
        special_tokens: list[str],
        num_processes: int = 4
) -> Counter[tuple[bytes, ...]]:
    """
    Run multiprocessing pre-tokenization on the chunks and store the counts for each pre-token.
    """

    if not special_tokens:
        with open(input_path, "r", encoding= "utf-8") as f:
            text = f.read()
        return pretokenize(text, special_tokens)
    
    split_special_token = special_tokens[0].encode("utf-8")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

    jobs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        jobs.append((input_path, start, end, special_tokens))
    
    with Pool(num_processes) as pool:
        counters = pool.starmap(count_pretokens_in_chunk, jobs)

    total_counts = Counter()
    for counter in counters:
        total_counts.update(counter)

    return total_counts







if __name__ == "__main__":
    input_path = "tiny_test.txt"

    with open(input_path, "w", encoding = "utf-8") as f:
        f.write("low low lower<|endoftext|>newer wider\nhello world! hello world!")

    vocab, merges = train_bpe(
        input_path,
        vocab_size=270,
        special_tokens=["<|endoftext|>"]
    )

    print("vocab size:", len(vocab))
    print("num merges:", len(merges))
    print("first 10 merges:", merges[:10])