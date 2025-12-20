from gpt2 import Tokenizer

def test_gpt2_tokenizer():
    tokenizer = Tokenizer('gpt2')
    test_string = "hello, world"
    encoded = tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
    decoded = tokenizer.decode(encoded)
    assert decoded == test_string