"""Tests for torch9.text module."""

import pytest


def test_tokenizer():
    """Test Tokenizer initialization."""
    from torch9.text import Tokenizer
    
    tokenizer = Tokenizer(vocab_size=10000, max_length=512)
    assert tokenizer.vocab_size == 10000
    assert tokenizer.max_length == 512


def test_tokenizer_encode():
    """Test Tokenizer encode method."""
    from torch9.text import Tokenizer
    
    tokenizer = Tokenizer()
    text = "Hello world"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)


def test_tokenizer_decode():
    """Test Tokenizer decode method."""
    from torch9.text import Tokenizer
    
    tokenizer = Tokenizer()
    tokens = [1, 2, 3]
    text = tokenizer.decode(tokens)
    assert isinstance(text, str)


def test_text_dataset():
    """Test TextDataset."""
    from torch9.text import TextDataset
    
    data = ["sentence 1", "sentence 2", "sentence 3"]
    dataset = TextDataset(data)
    assert len(dataset) == 3
    assert dataset[0] == "sentence 1"
    assert dataset[2] == "sentence 3"


def test_tokenize_function():
    """Test tokenize function."""
    from torch9.text import tokenize
    
    text = "Test tokenization"
    tokens = tokenize(text)
    assert isinstance(tokens, list)


def test_tokenize_with_custom_tokenizer():
    """Test tokenize with custom tokenizer."""
    from torch9.text import tokenize, Tokenizer
    
    tokenizer = Tokenizer(vocab_size=5000)
    text = "Custom tokenizer test"
    tokens = tokenize(text, tokenizer)
    assert isinstance(tokens, list)
