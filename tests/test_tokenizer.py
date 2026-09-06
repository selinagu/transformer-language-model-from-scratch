from cs336_basics.tokenizer import Tokenizer

def test_tokenizer():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=[],
        special_tokens=["<|endoftext|>"],
    )

    texts = [
        "",
        "hello world",
        "你好，世界",
        "emoji: 🧠",
        "line one\nline two",
    ]
    for t in texts:
        assert tokenizer.decode(tokenizer.encode(t)) == t

    ids = tokenizer.encode("hello<|endoftext|>world")
    assert ids.count(256) == 1

    text = "hello<|endoftext|>world 你好"
    chunks = ["hel", "lo<|end", "oftext|>wo", "rld ", "你好"]

    assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)