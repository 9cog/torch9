"""Tests for torch9.data module."""

import pytest


def test_data_pipeline():
    """Test DataPipeline initialization."""
    from torch9.data import DataPipeline
    
    source = [1, 2, 3, 4, 5]
    pipeline = DataPipeline(source)
    assert pipeline.source == source
    assert len(pipeline.transforms) == 0


def test_data_pipeline_map():
    """Test DataPipeline map operation."""
    from torch9.data import DataPipeline
    
    pipeline = DataPipeline([1, 2, 3])
    pipeline = pipeline.map(lambda x: x * 2)
    assert len(pipeline.transforms) == 1
    assert pipeline.transforms[0][0] == "map"


def test_data_pipeline_filter():
    """Test DataPipeline filter operation."""
    from torch9.data import DataPipeline
    
    pipeline = DataPipeline([1, 2, 3])
    pipeline = pipeline.filter(lambda x: x > 1)
    assert len(pipeline.transforms) == 1
    assert pipeline.transforms[0][0] == "filter"


def test_data_pipeline_batch():
    """Test DataPipeline batch operation."""
    from torch9.data import DataPipeline
    
    pipeline = DataPipeline([1, 2, 3])
    pipeline = pipeline.batch(2)
    assert len(pipeline.transforms) == 1
    assert pipeline.transforms[0] == ("batch", 2)


def test_data_pipeline_chaining():
    """Test DataPipeline operation chaining."""
    from torch9.data import DataPipeline
    
    pipeline = DataPipeline([1, 2, 3, 4, 5])
    pipeline = (pipeline
        .map(lambda x: x * 2)
        .filter(lambda x: x > 4)
        .batch(2))
    
    assert len(pipeline.transforms) == 3


def test_data_pipeline_iter():
    """Test DataPipeline iteration."""
    from torch9.data import DataPipeline
    
    pipeline = DataPipeline([1, 2, 3])
    items = list(pipeline)
    assert len(items) > 0


def test_data_loader():
    """Test DataLoader initialization."""
    from torch9.data import DataLoader
    
    dataset = [1, 2, 3, 4, 5]
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    assert loader.batch_size == 2
    assert loader.shuffle is True
    assert loader.num_workers == 0


def test_data_loader_iter():
    """Test DataLoader iteration."""
    from torch9.data import DataLoader
    
    dataset = list(range(10))
    loader = DataLoader(dataset, batch_size=3)
    
    batches = list(loader)
    assert len(batches) > 0


def test_data_loader_len():
    """Test DataLoader length."""
    from torch9.data import DataLoader
    
    dataset = list(range(10))
    loader = DataLoader(dataset, batch_size=3)
    
    assert loader.__len__() > 0
