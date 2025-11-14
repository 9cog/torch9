"""Shared test fixtures and configuration."""

import pytest
import torch


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(10, 3, 224, 224)


@pytest.fixture
def sample_waveform():
    """Create a sample audio waveform for testing."""
    return torch.randn(1, 16000)


@pytest.fixture
def sample_text():
    """Create sample text for testing."""
    return "This is a sample text for testing purposes"


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return list(range(100))
