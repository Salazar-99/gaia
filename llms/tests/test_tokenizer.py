from llms import Tokenizer

def test_gpt2_tokenizer():
    tokenizer = Tokenizer('gpt2')
    test_string = "hello, world"
    encoded = tokenizer.encode(test_string)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_string