"""Tests for torch9 main package."""

import pytest


def test_import_torch9():
    """Test that torch9 can be imported."""
    import torch9
    assert torch9.__version__ == "0.1.0"


def test_lazy_import_audio():
    """Test lazy import of audio module."""
    import torch9
    audio = torch9.audio
    assert hasattr(audio, 'load_audio')
    assert hasattr(audio, 'save_audio')
    assert hasattr(audio, 'AudioTransform')


def test_lazy_import_vision():
    """Test lazy import of vision module."""
    import torch9
    vision = torch9.vision
    assert hasattr(vision, 'load_image')
    assert hasattr(vision, 'ImageTransform')
    assert hasattr(vision, 'ResNet')


def test_lazy_import_text():
    """Test lazy import of text module."""
    import torch9
    text = torch9.text
    assert hasattr(text, 'Tokenizer')
    assert hasattr(text, 'TextDataset')
    assert hasattr(text, 'tokenize')


def test_lazy_import_rl():
    """Test lazy import of rl module."""
    import torch9
    rl = torch9.rl
    assert hasattr(rl, 'Environment')
    assert hasattr(rl, 'Policy')
    assert hasattr(rl, 'Agent')


def test_lazy_import_rec():
    """Test lazy import of rec module."""
    import torch9
    rec = torch9.rec
    assert hasattr(rec, 'EmbeddingTable')
    assert hasattr(rec, 'RecommenderModel')


def test_lazy_import_tune():
    """Test lazy import of tune module."""
    import torch9
    tune = torch9.tune
    assert hasattr(tune, 'FineTuner')
    assert hasattr(tune, 'LoRAConfig')


def test_lazy_import_data():
    """Test lazy import of data module."""
    import torch9
    data = torch9.data
    assert hasattr(data, 'DataPipeline')
    assert hasattr(data, 'DataLoader')


def test_lazy_import_codec():
    """Test lazy import of codec module."""
    import torch9
    codec = torch9.codec
    assert hasattr(codec, 'VideoDecoder')
    assert hasattr(codec, 'AudioDecoder')
    assert hasattr(codec, 'decode_video')
    assert hasattr(codec, 'decode_audio')


def test_all_modules_in_all():
    """Test that __all__ contains all expected modules."""
    import torch9
    expected_modules = [
        'audio', 'vision', 'text', 'rl',
        'rec', 'tune', 'data', 'codec'
    ]
    for module in expected_modules:
        assert module in torch9.__all__


def test_invalid_attribute():
    """Test that accessing invalid attribute raises AttributeError."""
    import torch9
    with pytest.raises(AttributeError):
        _ = torch9.nonexistent_module
