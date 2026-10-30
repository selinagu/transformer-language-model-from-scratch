import time
import tracemalloc

from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import save_tokenizer_artifacts


def main():
    input_path = "data/owt_train.txt"  
    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]

    tracemalloc.start()
    start_time = time.time()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    vocab_path = "outputs/openwebtext_vocab_32000.pkl"
    merges_path = "outputs/openwebtext_merges_32000.pkl"

    save_tokenizer_artifacts(vocab, merges, vocab_path, merges_path)

    longest_token_id, longest_token = max(vocab.items(), key=lambda x: len(x[1]))

    print("vocab size:", len(vocab))
    print("num merges:", len(merges))
    print("vocabulary artifact:", vocab_path)
    print("merges artifact:", merges_path)

    print("training time minutes:", elapsed / 60)
    print("peak memory MB:", peak_mem / 1024 / 1024)

    print("longest token id:", longest_token_id)
    print("longest token length in bytes:", len(longest_token))
    print(f"longest token bytes: {longest_token}")
    print("longest token decoded:", longest_token.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()