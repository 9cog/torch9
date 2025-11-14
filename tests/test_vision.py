"""Tests for torch9.vision module."""

import pytest
import torch


def test_image_transform():
    """Test ImageTransform initialization."""
    from torch9.vision import ImageTransform
    
    transform = ImageTransform(size=(224, 224))
    assert transform.size == (224, 224)


def test_image_transform_call():
    """Test ImageTransform __call__ method."""
    from torch9.vision import ImageTransform
    
    transform = ImageTransform()
    image = torch.randn(3, 224, 224)
    result = transform(image)
    assert result.shape == image.shape


def test_resnet():
    """Test ResNet model initialization."""
    from torch9.vision import ResNet
    
    model = ResNet(num_layers=50, pretrained=False)
    assert model.num_layers == 50
    assert model.pretrained is False


def test_resnet_call():
    """Test ResNet forward pass."""
    from torch9.vision import ResNet
    
    model = ResNet(num_layers=18)
    x = torch.randn(1, 3, 224, 224)
    result = model(x)
    assert result.shape == x.shape


def test_load_image():
    """Test load_image function."""
    from torch9.vision import load_image
    
    image = load_image("test.jpg")
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3  # RGB channels


def test_load_image_with_size():
    """Test load_image with specific size."""
    from torch9.vision import load_image
    
    image = load_image("test.jpg", size=(128, 128))
    assert image.shape == (3, 128, 128)
