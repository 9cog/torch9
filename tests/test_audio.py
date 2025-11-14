"""Tests for torch9.audio module."""

import pytest
import torch


def test_audio_transform():
    """Test AudioTransform initialization."""
    from torch9.audio import AudioTransform

    transform = AudioTransform(sample_rate=16000)
    assert transform.sample_rate == 16000

    # Test with different sample rate
    transform2 = AudioTransform(sample_rate=22050)
    assert transform2.sample_rate == 22050


def test_audio_transform_call():
    """Test AudioTransform __call__ method."""
    from torch9.audio import AudioTransform

    transform = AudioTransform()
    waveform = torch.randn(1, 16000)
    result = transform(waveform)
    assert result.shape == waveform.shape


def test_load_audio():
    """Test load_audio function."""
    from torch9.audio import load_audio

    waveform, sample_rate = load_audio("test.wav")
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(sample_rate, int)
    assert sample_rate == 16000


def test_load_audio_with_sample_rate():
    """Test load_audio with specific sample rate."""
    from torch9.audio import load_audio

    waveform, sample_rate = load_audio("test.wav", sample_rate=22050)
    assert sample_rate == 22050


def test_save_audio():
    """Test save_audio function."""
    from torch9.audio import save_audio

    waveform = torch.randn(1, 16000)
    # Should not raise exception
    save_audio("output.wav", waveform, sample_rate=16000)
