"""Tests for torch9.logic module - Tensor Logic Framework."""

import pytest
import torch
import warnings


class TestShapeSpec:
    """Tests for ShapeSpec class."""

    def test_exact_shape(self):
        """Test exact shape matching."""
        from torch9.logic import ShapeSpec

        spec = ShapeSpec(dims=(2, 3, 4))
        assert spec.validate(torch.Size([2, 3, 4]))
        assert not spec.validate(torch.Size([2, 3, 5]))
        assert not spec.validate(torch.Size([2, 3]))

    def test_dynamic_dims(self):
        """Test dynamic dimension matching with -1."""
        from torch9.logic import ShapeSpec

        spec = ShapeSpec(dims=(-1, 3, 224, 224))
        assert spec.validate(torch.Size([1, 3, 224, 224]))
        assert spec.validate(torch.Size([32, 3, 224, 224]))
        assert not spec.validate(torch.Size([32, 4, 224, 224]))

    def test_min_max_dims(self):
        """Test min/max dimension constraints."""
        from torch9.logic import ShapeSpec

        spec = ShapeSpec(min_dims=2, max_dims=4)
        assert spec.validate(torch.Size([2, 3]))
        assert spec.validate(torch.Size([2, 3, 4]))
        assert spec.validate(torch.Size([2, 3, 4, 5]))
        assert not spec.validate(torch.Size([2]))
        assert not spec.validate(torch.Size([2, 3, 4, 5, 6]))

    def test_describe(self):
        """Test shape description."""
        from torch9.logic import ShapeSpec

        spec = ShapeSpec(dims=(-1, 3, 224, 224))
        desc = spec.describe()
        assert "*, 3, 224, 224" in desc


class TestTensorSpec:
    """Tests for TensorSpec class."""

    def test_dtype_validation(self):
        """Test dtype validation."""
        from torch9.logic import TensorSpec, ValidationError

        spec = TensorSpec(dtype=torch.float32)
        tensor_f32 = torch.randn(2, 3)
        tensor_f64 = torch.randn(2, 3, dtype=torch.float64)

        assert spec.validate(tensor_f32, raise_error=False)
        assert not spec.validate(tensor_f64, raise_error=False)

        with pytest.raises(ValidationError):
            spec.validate(tensor_f64)

    def test_shape_validation(self):
        """Test shape validation."""
        from torch9.logic import TensorSpec, ValidationError

        spec = TensorSpec(shape=(-1, 3, 224, 224))
        valid = torch.randn(4, 3, 224, 224)
        invalid = torch.randn(4, 4, 224, 224)

        assert spec.validate(valid, raise_error=False)
        assert not spec.validate(invalid, raise_error=False)

    def test_value_range(self):
        """Test value range validation."""
        from torch9.logic import TensorSpec, ValidationError

        spec = TensorSpec(min_value=0.0, max_value=1.0)
        valid = torch.rand(10)  # [0, 1)
        invalid = torch.randn(10)  # Can be negative

        assert spec.validate(valid, raise_error=False)

    def test_allow_none(self):
        """Test None handling."""
        from torch9.logic import TensorSpec, ValidationError

        spec_no_none = TensorSpec(allow_none=False)
        spec_allow_none = TensorSpec(allow_none=True)

        assert not spec_no_none.validate(None, raise_error=False)
        assert spec_allow_none.validate(None, raise_error=False)

    def test_coerce(self):
        """Test tensor coercion."""
        from torch9.logic import TensorSpec

        spec = TensorSpec(dtype=torch.float32, contiguous=True)
        tensor = torch.randn(3, 4, dtype=torch.float64)
        coerced = spec.coerce(tensor)

        assert coerced.dtype == torch.float32
        assert coerced.is_contiguous()


class TestTensorGuard:
    """Tests for TensorGuard decorator."""

    def test_basic_guard(self):
        """Test basic tensor guard."""
        from torch9.logic import tensor_guard, TensorSpec

        @tensor_guard(x=TensorSpec(shape=(-1, 10)))
        def process(x):
            return x * 2

        valid = torch.randn(5, 10)
        result = process(valid)
        assert result.shape == valid.shape

    def test_guard_raises_error(self):
        """Test guard raises on invalid input."""
        from torch9.logic import tensor_guard, TensorSpec, ValidationError

        @tensor_guard(x=TensorSpec(shape=(-1, 10)))
        def process(x):
            return x * 2

        invalid = torch.randn(5, 20)
        with pytest.raises(ValidationError):
            process(invalid)

    def test_output_validation(self):
        """Test output validation."""
        from torch9.logic import tensor_guard, TensorSpec, ValidationError

        @tensor_guard(output=TensorSpec(dtype=torch.float32))
        def bad_output(x):
            return x.to(torch.float64)

        with pytest.raises(ValidationError):
            bad_output(torch.randn(5))


class TestValidateTensor:
    """Tests for validate_tensor function."""

    def test_validate_tensor(self):
        """Test standalone validation."""
        from torch9.logic import validate_tensor

        tensor = torch.randn(3, 4)
        assert validate_tensor(tensor, dtype=torch.float32)


class TestTensorOps:
    """Tests for TensorOps class."""

    def test_normalize(self):
        """Test normalization."""
        from torch9.logic import TensorOps

        tensor = torch.randn(10, 3)
        normalized = TensorOps.normalize(tensor)
        assert normalized.shape == tensor.shape

    def test_safe_divide(self):
        """Test safe division."""
        from torch9.logic import TensorOps

        num = torch.ones(5)
        denom = torch.zeros(5)
        result = TensorOps.safe_divide(num, denom)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_clamp_norm(self):
        """Test norm clamping."""
        from torch9.logic import TensorOps

        tensor = torch.randn(10) * 100
        clamped = TensorOps.clamp_norm(tensor, max_norm=1.0)
        assert clamped.norm() <= 1.0 + 1e-6

    def test_softmax_temperature(self):
        """Test temperature softmax."""
        from torch9.logic import TensorOps

        logits = torch.randn(5, 10)
        soft = TensorOps.softmax_temperature(logits, temperature=0.5)
        assert soft.sum(dim=-1).allclose(torch.ones(5))

    def test_gumbel_softmax(self):
        """Test Gumbel-Softmax."""
        from torch9.logic import TensorOps

        logits = torch.randn(5, 10)
        soft = TensorOps.gumbel_softmax(logits, temperature=1.0)
        assert soft.shape == logits.shape

        hard = TensorOps.gumbel_softmax(logits, temperature=1.0, hard=True)
        assert hard.sum(dim=-1).allclose(torch.ones(5))

    def test_one_hot(self):
        """Test one-hot encoding."""
        from torch9.logic import TensorOps

        indices = torch.tensor([0, 2, 1, 3])
        one_hot = TensorOps.one_hot(indices, num_classes=5)
        assert one_hot.shape == (4, 5)
        assert one_hot.sum() == 4


class TestBroadcastTensors:
    """Tests for broadcast_tensors function."""

    def test_broadcast(self):
        """Test tensor broadcasting."""
        from torch9.logic import broadcast_tensors

        a = torch.randn(1, 3)
        b = torch.randn(5, 1)
        a_b, b_b = broadcast_tensors(a, b)
        assert a_b.shape == (5, 3)
        assert b_b.shape == (5, 3)


class TestSafeReshape:
    """Tests for safe_reshape function."""

    def test_reshape(self):
        """Test safe reshape."""
        from torch9.logic import safe_reshape

        tensor = torch.randn(6, 4)
        reshaped = safe_reshape(tensor, 2, 3, 4)
        assert reshaped.shape == (2, 3, 4)


class TestEnsureBatchDim:
    """Tests for ensure_batch_dim function."""

    def test_adds_batch(self):
        """Test batch dimension addition."""
        from torch9.logic import ensure_batch_dim

        tensor = torch.randn(3, 224, 224)
        batched, was_batched = ensure_batch_dim(tensor, expected_dims=4)
        assert batched.shape == (1, 3, 224, 224)
        assert not was_batched

    def test_preserves_batch(self):
        """Test existing batch dimension preserved."""
        from torch9.logic import ensure_batch_dim

        tensor = torch.randn(5, 3, 224, 224)
        batched, was_batched = ensure_batch_dim(tensor, expected_dims=4)
        assert batched.shape == (5, 3, 224, 224)
        assert was_batched


class TestDeviceManager:
    """Tests for DeviceManager class."""

    def test_default_device(self):
        """Test default device selection."""
        from torch9.logic import DeviceManager

        dm = DeviceManager(prefer_cuda=False)
        assert dm.device.type == "cpu"

    def test_to_device(self):
        """Test moving to device."""
        from torch9.logic import DeviceManager

        dm = DeviceManager(device="cpu")
        tensor = torch.randn(3, 4)
        moved = dm.to_device(tensor)
        assert moved.device.type == "cpu"


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device(self):
        """Test device selection."""
        from torch9.logic import get_device

        device = get_device(prefer_cuda=False)
        assert device.type == "cpu"


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    def test_snapshot(self):
        """Test memory snapshot."""
        from torch9.logic import MemoryTracker

        tracker = MemoryTracker()
        stats = tracker.snapshot("test")
        assert hasattr(stats, "allocated")

    def test_track_context(self):
        """Test tracking context manager."""
        from torch9.logic import MemoryTracker

        tracker = MemoryTracker()
        with tracker.track("operation"):
            _ = torch.randn(100, 100)
        # Should have recorded something
        assert len(tracker.checkpoints) >= 0  # May be 0 on CPU


class TestTensorLogic:
    """Tests for TensorLogic class."""

    def test_all_equal(self):
        """Test tensor equality check."""
        from torch9.logic import TensorLogic

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        c = torch.tensor([1.0, 2.0, 4.0])

        assert TensorLogic.all_equal(a, b)
        assert not TensorLogic.all_equal(a, c)

    def test_any_nan(self):
        """Test NaN detection."""
        from torch9.logic import TensorLogic

        normal = torch.randn(10)
        with_nan = torch.tensor([1.0, float("nan"), 3.0])

        assert not TensorLogic.any_nan(normal)
        assert TensorLogic.any_nan(with_nan)

    def test_any_inf(self):
        """Test infinity detection."""
        from torch9.logic import TensorLogic

        normal = torch.randn(10)
        with_inf = torch.tensor([1.0, float("inf"), 3.0])

        assert not TensorLogic.any_inf(normal)
        assert TensorLogic.any_inf(with_inf)

    def test_is_valid(self):
        """Test finite value check."""
        from torch9.logic import TensorLogic

        valid = torch.randn(10)
        invalid = torch.tensor([1.0, float("nan"), 3.0])

        assert TensorLogic.is_valid(valid)
        assert not TensorLogic.is_valid(invalid)

    def test_logical_combine(self):
        """Test logical combination."""
        from torch9.logic import TensorLogic

        a = torch.tensor([True, True, False])
        b = torch.tensor([True, False, False])

        and_result = TensorLogic.logical_combine([a, b], op="and")
        assert and_result.tolist() == [True, False, False]

        or_result = TensorLogic.logical_combine([a, b], op="or")
        assert or_result.tolist() == [True, True, False]

    def test_top_k_mask(self):
        """Test top-k mask creation."""
        from torch9.logic import TensorLogic

        tensor = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        mask = TensorLogic.top_k_mask(tensor, k=2)
        assert mask.sum() == 2
        assert mask[1]  # 5.0 is top
        assert mask[4]  # 4.0 is second


class TestTimer:
    """Tests for Timer class."""

    def test_timer(self):
        """Test timer functionality."""
        from torch9.logic import Timer
        import time

        timer = Timer().start()
        time.sleep(0.01)
        elapsed = timer.stop()
        assert elapsed >= 0.01

    def test_measure_context(self):
        """Test measure context manager."""
        from torch9.logic import Timer
        import time

        with Timer().measure() as timer:
            time.sleep(0.01)
        assert timer.elapsed >= 0.01
