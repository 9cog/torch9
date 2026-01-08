"""Tests for torch9.audio module."""

import pytest
import torch
import warnings


class TestAudioTransform:
    """Tests for AudioTransform class."""

    def test_init_default(self):
        """Test AudioTransform initialization with defaults."""
        from torch9.audio import AudioTransform

        transform = AudioTransform()
        assert transform.sample_rate == 16000

    def test_init_custom_sample_rate(self):
        """Test AudioTransform initialization with custom sample rate."""
        from torch9.audio import AudioTransform

        transform = AudioTransform(sample_rate=22050)
        assert transform.sample_rate == 22050

    def test_call_returns_input(self):
        """Test AudioTransform __call__ returns input unchanged."""
        from torch9.audio import AudioTransform

        transform = AudioTransform()
        waveform = torch.randn(1, 16000)
        result = transform(waveform)
        assert torch.equal(result, waveform)

    def test_repr(self):
        """Test AudioTransform string representation."""
        from torch9.audio import AudioTransform

        transform = AudioTransform(sample_rate=16000)
        assert "AudioTransform" in repr(transform)
        assert "16000" in repr(transform)


class TestMelSpectrogram:
    """Tests for MelSpectrogram class."""

    def test_init(self):
        """Test MelSpectrogram initialization."""
        from torch9.audio import MelSpectrogram

        mel = MelSpectrogram(sample_rate=16000, n_mels=80)
        assert mel.sample_rate == 16000
        assert mel.n_mels == 80

    def test_call_returns_tensor(self):
        """Test MelSpectrogram returns tensor."""
        from torch9.audio import MelSpectrogram

        mel = MelSpectrogram(sample_rate=16000, n_mels=80)
        waveform = torch.randn(1, 16000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mel(waveform)
        assert isinstance(result, torch.Tensor)


class TestMFCC:
    """Tests for MFCC class."""

    def test_init(self):
        """Test MFCC initialization."""
        from torch9.audio import MFCC

        mfcc = MFCC(sample_rate=16000, n_mfcc=40)
        assert mfcc.sample_rate == 16000
        assert mfcc.n_mfcc == 40

    def test_call_returns_tensor(self):
        """Test MFCC returns tensor."""
        from torch9.audio import MFCC

        mfcc = MFCC(sample_rate=16000, n_mfcc=40)
        waveform = torch.randn(1, 16000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = mfcc(waveform)
        assert isinstance(result, torch.Tensor)


class TestResample:
    """Tests for Resample class."""

    def test_init(self):
        """Test Resample initialization."""
        from torch9.audio import Resample

        resampler = Resample(orig_freq=16000, new_freq=8000)
        assert resampler.orig_freq == 16000
        assert resampler.new_freq == 8000

    def test_call_changes_length(self):
        """Test Resample changes waveform length."""
        from torch9.audio import Resample

        resampler = Resample(orig_freq=16000, new_freq=8000)
        waveform = torch.randn(1, 16000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = resampler(waveform)
        # Should be approximately half the length
        assert result.shape[-1] < waveform.shape[-1]


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_returns_tuple(self):
        """Test load_audio returns tuple of (waveform, sample_rate)."""
        from torch9.audio import load_audio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sample_rate = load_audio("test.wav")
        assert isinstance(waveform, torch.Tensor)
        assert isinstance(sample_rate, int)

    def test_with_sample_rate(self):
        """Test load_audio with specific sample rate."""
        from torch9.audio import load_audio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sample_rate = load_audio("test.wav", sample_rate=22050)
        # In fallback mode, should return the requested sample rate
        assert sample_rate == 22050

    def test_default_sample_rate(self):
        """Test load_audio default sample rate."""
        from torch9.audio import load_audio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sample_rate = load_audio("test.wav")
        assert sample_rate == 16000


class TestSaveAudio:
    """Tests for save_audio function."""

    def test_no_exception(self):
        """Test save_audio doesn't raise exception."""
        from torch9.audio import save_audio

        waveform = torch.randn(1, 16000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Should not raise exception even in fallback mode
            save_audio("output.wav", waveform, sample_rate=16000)


class TestUtilities:
    """Tests for utility functions."""

    def test_is_torchaudio_available(self):
        """Test is_torchaudio_available returns boolean."""
        from torch9.audio import is_torchaudio_available

        result = is_torchaudio_available()
        assert isinstance(result, bool)
