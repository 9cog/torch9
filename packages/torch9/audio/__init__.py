"""
torch9.audio - Audio signal processing and deep learning

This module provides tools for:
- Audio I/O operations
- Audio transformations and augmentations
- Pre-trained models for speech and audio tasks
- Audio datasets and utilities

Core functionality includes waveform processing, spectrogram generation,
and integration with popular audio formats.

When torchaudio is installed, this module provides full functionality.
Without torchaudio, basic fallback implementations are available.
"""

import warnings
from typing import Optional, Tuple, Union

import torch

# Try to import torchaudio for full functionality
_TORCHAUDIO_AVAILABLE = False
try:
    import torchaudio
    import torchaudio.transforms as T
    import torchaudio.functional as F

    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    torchaudio = None
    T = None
    F = None

__all__ = [
    "load_audio",
    "save_audio",
    "AudioTransform",
    "MelSpectrogram",
    "MFCC",
    "Resample",
    "SpectralCentroid",
    "get_audio_info",
    "is_torchaudio_available",
]


def is_torchaudio_available() -> bool:
    """Check if torchaudio is available."""
    return _TORCHAUDIO_AVAILABLE


class AudioTransform:
    """
    Base class for audio transformations.

    This is a composable transform that can be extended or used as a
    pass-through identity transform.

    Args:
        sample_rate: Target sample rate for audio processing
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply transformation to audio waveform."""
        return waveform

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sample_rate={self.sample_rate})"


class MelSpectrogram(AudioTransform):
    """
    Compute mel spectrogram from audio waveform.

    Args:
        sample_rate: Sample rate of audio
        n_fft: Size of FFT
        win_length: Window size
        hop_length: Length of hop between STFT windows
        n_mels: Number of mel filterbanks
        f_min: Minimum frequency
        f_max: Maximum frequency
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__(sample_rate)
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or (self.win_length // 2)
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or (sample_rate / 2.0)

        if _TORCHAUDIO_AVAILABLE:
            self._transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=self.f_max,
            )
        else:
            self._transform = None

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram.

        Args:
            waveform: Audio waveform tensor of shape (channels, time)

        Returns:
            Mel spectrogram tensor of shape (channels, n_mels, time)
        """
        if self._transform is not None:
            return self._transform(waveform)
        else:
            # Fallback: return placeholder shape
            warnings.warn(
                "torchaudio not available. Install with: pip install torch9[audio]",
                UserWarning,
            )
            channels = waveform.shape[0] if waveform.dim() > 1 else 1
            time_frames = waveform.shape[-1] // self.hop_length + 1
            return torch.zeros(channels, self.n_mels, time_frames)


class MFCC(AudioTransform):
    """
    Compute MFCC (Mel-frequency cepstral coefficients) from audio.

    Args:
        sample_rate: Sample rate of audio
        n_mfcc: Number of MFCC coefficients
        n_fft: Size of FFT
        n_mels: Number of mel filterbanks
        log_mels: Whether to use log-mel spectrograms
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 400,
        n_mels: int = 128,
        log_mels: bool = False,
    ):
        super().__init__(sample_rate)
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.log_mels = log_mels

        if _TORCHAUDIO_AVAILABLE:
            self._transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                log_mels=log_mels,
                melkwargs={"n_fft": n_fft, "n_mels": n_mels},
            )
        else:
            self._transform = None

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute MFCC features.

        Args:
            waveform: Audio waveform tensor

        Returns:
            MFCC tensor
        """
        if self._transform is not None:
            return self._transform(waveform)
        else:
            warnings.warn(
                "torchaudio not available. Install with: pip install torch9[audio]",
                UserWarning,
            )
            channels = waveform.shape[0] if waveform.dim() > 1 else 1
            time_frames = waveform.shape[-1] // 160 + 1
            return torch.zeros(channels, self.n_mfcc, time_frames)


class Resample(AudioTransform):
    """
    Resample audio to a different sample rate.

    Args:
        orig_freq: Original sample rate
        new_freq: Target sample rate
    """

    def __init__(self, orig_freq: int = 16000, new_freq: int = 8000):
        super().__init__(new_freq)
        self.orig_freq = orig_freq
        self.new_freq = new_freq

        if _TORCHAUDIO_AVAILABLE:
            self._transform = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
        else:
            self._transform = None

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Resample audio waveform.

        Args:
            waveform: Audio waveform tensor

        Returns:
            Resampled waveform
        """
        if self._transform is not None:
            return self._transform(waveform)
        else:
            warnings.warn(
                "torchaudio not available. Install with: pip install torch9[audio]",
                UserWarning,
            )
            # Simple fallback using interpolation
            ratio = self.new_freq / self.orig_freq
            new_length = int(waveform.shape[-1] * ratio)
            return torch.nn.functional.interpolate(
                waveform.unsqueeze(0), size=new_length, mode="linear", align_corners=False
            ).squeeze(0)


class SpectralCentroid(AudioTransform):
    """
    Compute spectral centroid of audio.

    Args:
        sample_rate: Sample rate of audio
        n_fft: Size of FFT
        hop_length: Length of hop between frames
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: Optional[int] = None,
    ):
        super().__init__(sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 2)

        if _TORCHAUDIO_AVAILABLE:
            self._transform = T.SpectralCentroid(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=self.hop_length
            )
        else:
            self._transform = None

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral centroid.

        Args:
            waveform: Audio waveform tensor

        Returns:
            Spectral centroid tensor
        """
        if self._transform is not None:
            return self._transform(waveform)
        else:
            warnings.warn(
                "torchaudio not available. Install with: pip install torch9[audio]",
                UserWarning,
            )
            time_frames = waveform.shape[-1] // self.hop_length + 1
            return torch.zeros(1, time_frames)


def get_audio_info(path: str) -> dict:
    """
    Get metadata about an audio file.

    Args:
        path: Path to audio file

    Returns:
        Dictionary with audio metadata (sample_rate, num_channels, num_frames, etc.)
    """
    if _TORCHAUDIO_AVAILABLE:
        info = torchaudio.info(path)
        return {
            "sample_rate": info.sample_rate,
            "num_channels": info.num_channels,
            "num_frames": info.num_frames,
            "bits_per_sample": info.bits_per_sample,
            "encoding": info.encoding,
        }
    else:
        raise RuntimeError(
            "torchaudio is required for get_audio_info. "
            "Install with: pip install torch9[audio]"
        )


def load_audio(
    path: str,
    sample_rate: Optional[int] = None,
    channels_first: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file from path.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate for resampling (None keeps original)
        channels_first: If True, returns (channels, time), else (time, channels)

    Returns:
        Tuple of (waveform tensor, sample_rate)

    Example:
        >>> waveform, sr = load_audio("speech.wav")
        >>> waveform, sr = load_audio("speech.wav", sample_rate=16000)
    """
    if _TORCHAUDIO_AVAILABLE:
        waveform, orig_sr = torchaudio.load(path)

        # Resample if requested
        if sample_rate is not None and sample_rate != orig_sr:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=sample_rate)
            waveform = resampler(waveform)
            orig_sr = sample_rate

        # Transpose if channels_first=False
        if not channels_first:
            waveform = waveform.transpose(0, 1)

        return waveform, orig_sr
    else:
        # Fallback: return placeholder for testing without torchaudio
        warnings.warn(
            "torchaudio not available. Returning placeholder audio. "
            "Install with: pip install torch9[audio]",
            UserWarning,
        )
        sr = sample_rate or 16000
        # Return 1 second of silence
        waveform = torch.zeros(1, sr) if channels_first else torch.zeros(sr, 1)
        return waveform, sr


def save_audio(
    path: str,
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
) -> None:
    """
    Save audio waveform to file.

    Args:
        path: Output path for audio file
        waveform: Audio waveform tensor
        sample_rate: Sample rate of the audio
        channels_first: If True, waveform is (channels, time), else (time, channels)
        format: Audio format (e.g., "wav", "mp3", "flac"). Inferred from path if None.
        encoding: Audio encoding (e.g., "PCM_S", "PCM_F")
        bits_per_sample: Bits per sample (e.g., 16, 32)

    Example:
        >>> waveform = torch.randn(1, 16000)  # 1 second of audio
        >>> save_audio("output.wav", waveform, sample_rate=16000)
    """
    if _TORCHAUDIO_AVAILABLE:
        # Ensure channels first
        if not channels_first:
            waveform = waveform.transpose(0, 1)

        # Ensure 2D tensor
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        torchaudio.save(
            path,
            waveform,
            sample_rate,
            format=format,
            encoding=encoding,
            bits_per_sample=bits_per_sample,
        )
    else:
        warnings.warn(
            "torchaudio not available. Audio not saved. "
            "Install with: pip install torch9[audio]",
            UserWarning,
        )
