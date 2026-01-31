"""Tests for torch9.luatorch - Unified LuaTorch Core."""

import math
import os
import tempfile

import pytest
import torch


# =============================================================================
# Tensor type constructors
# =============================================================================


class TestTensorTypes:
    """Tests for typed tensor constructors."""

    def test_float_tensor_empty(self):
        from torch9.luatorch import FloatTensor

        t = FloatTensor()
        assert t.dtype == torch.float32

    def test_float_tensor_1d(self):
        from torch9.luatorch import FloatTensor

        t = FloatTensor(5)
        assert t.shape == (5,)
        assert t.dtype == torch.float32

    def test_float_tensor_2d(self):
        from torch9.luatorch import FloatTensor

        t = FloatTensor(3, 4)
        assert t.shape == (3, 4)
        assert t.dtype == torch.float32

    def test_float_tensor_from_list(self):
        from torch9.luatorch import FloatTensor

        t = FloatTensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))

    def test_float_tensor_from_tensor(self):
        from torch9.luatorch import FloatTensor

        src = torch.tensor([1.0, 2.0], dtype=torch.float64)
        t = FloatTensor(src)
        assert t.dtype == torch.float32

    def test_double_tensor(self):
        from torch9.luatorch import DoubleTensor

        t = DoubleTensor(2, 3)
        assert t.dtype == torch.float64

    def test_long_tensor(self):
        from torch9.luatorch import LongTensor

        t = LongTensor(4)
        assert t.dtype == torch.int64

    def test_byte_tensor(self):
        from torch9.luatorch import ByteTensor

        t = ByteTensor(3)
        assert t.dtype == torch.uint8

    def test_int_tensor(self):
        from torch9.luatorch import IntTensor

        t = IntTensor(2, 2)
        assert t.dtype == torch.int32

    def test_half_tensor(self):
        from torch9.luatorch import HalfTensor

        t = HalfTensor(3)
        assert t.dtype == torch.float16

    def test_short_tensor(self):
        from torch9.luatorch import ShortTensor

        t = ShortTensor(3)
        assert t.dtype == torch.int16

    def test_char_tensor(self):
        from torch9.luatorch import CharTensor

        t = CharTensor(3)
        assert t.dtype == torch.int8


class TestDefaultTensorType:
    """Tests for default tensor type management."""

    def test_get_default(self):
        from torch9.luatorch import getdefaulttensortype

        assert getdefaulttensortype() == "torch.FloatTensor"

    def test_set_default(self):
        from torch9.luatorch import getdefaulttensortype, setdefaulttensortype

        original = getdefaulttensortype()
        try:
            setdefaulttensortype("torch.DoubleTensor")
            assert getdefaulttensortype() == "torch.DoubleTensor"
        finally:
            setdefaulttensortype(original)

    def test_set_invalid_type(self):
        from torch9.luatorch import setdefaulttensortype

        with pytest.raises(ValueError, match="Unknown tensor type"):
            setdefaulttensortype("torch.BogusType")


# =============================================================================
# Storage
# =============================================================================


class TestStorage:
    """Tests for the Storage wrapper."""

    def test_create_from_size(self):
        from torch9.luatorch import Storage

        s = Storage(10)
        assert s.size() == 10
        assert len(s) == 10

    def test_create_from_list(self):
        from torch9.luatorch import Storage

        s = Storage([1.0, 2.0, 3.0])
        assert s.size() == 3
        assert s[0] == 1.0
        assert s[2] == 3.0

    def test_create_from_tensor(self):
        from torch9.luatorch import Storage

        t = torch.tensor([5.0, 6.0])
        s = Storage(t)
        assert s.size() == 2
        assert s[0] == 5.0

    def test_fill(self):
        from torch9.luatorch import Storage

        s = Storage(5)
        s.fill(7.0)
        assert all(s[i] == 7.0 for i in range(5))

    def test_setitem(self):
        from torch9.luatorch import Storage

        s = Storage(3)
        s[0] = 42.0
        assert s[0] == 42.0

    def test_resize_larger(self):
        from torch9.luatorch import Storage

        s = Storage([1.0, 2.0])
        s.resize(5)
        assert s.size() == 5
        assert s[0] == 1.0

    def test_resize_smaller(self):
        from torch9.luatorch import Storage

        s = Storage([1.0, 2.0, 3.0])
        s.resize(2)
        assert s.size() == 2

    def test_tolist(self):
        from torch9.luatorch import Storage

        s = Storage([1.0, 2.0, 3.0])
        assert s.tolist() == [1.0, 2.0, 3.0]

    def test_copy(self):
        from torch9.luatorch import Storage

        a = Storage([1.0, 2.0])
        b = Storage(2)
        b.copy(a)
        assert b[0] == 1.0
        assert b[1] == 2.0

    def test_repr(self):
        from torch9.luatorch import Storage

        s = Storage(3)
        assert "Storage" in repr(s)
        assert "size=3" in repr(s)


# =============================================================================
# Class Registry
# =============================================================================


class TestClassRegistry:
    """Tests for the Torch7-style class registration system."""

    def test_typename_registered(self):
        from torch9.luatorch import nn, typename

        m = nn.Linear(10, 5)
        assert typename(m) == "nn.Linear"

    def test_typename_unregistered(self):
        from torch9.luatorch import typename

        assert "int" in typename(42)

    def test_isTypeOf_by_name(self):
        from torch9.luatorch import isTypeOf, nn

        m = nn.Linear(10, 5)
        assert isTypeOf(m, "nn.Linear")
        assert isTypeOf(m, "nn.Module")
        assert not isTypeOf(m, "nn.ReLU")

    def test_isTypeOf_by_class(self):
        from torch9.luatorch import isTypeOf, nn

        m = nn.Sequential()
        assert isTypeOf(m, nn.Container)
        assert isTypeOf(m, nn.Module)

    def test_registry_names(self):
        from torch9.luatorch import class_registry

        names = class_registry.registered_names
        assert "nn.Linear" in names
        assert "nn.Sequential" in names
        assert "nn.MSECriterion" in names

    def test_registry_get(self):
        from torch9.luatorch import class_registry, nn

        cls = class_registry.get("nn.ReLU")
        assert cls is nn.ReLU


# =============================================================================
# nn.Module base
# =============================================================================


class TestModule:
    """Tests for the base Module class."""

    def test_training_evaluate(self):
        from torch9.luatorch import nn

        m = nn.Module()
        assert m.train_mode is True
        m.evaluate()
        assert m.train_mode is False
        m.training()
        assert m.train_mode is True

    def test_clone(self):
        from torch9.luatorch import nn

        linear = nn.Linear(10, 5)
        cloned = linear.clone()
        assert cloned is not linear
        assert cloned.weight is not linear.weight
        assert cloned.weight.shape == linear.weight.shape


# =============================================================================
# nn.Linear
# =============================================================================


class TestLinear:
    """Tests for the Linear layer."""

    def test_output_shape(self):
        from torch9.luatorch import nn, FloatTensor

        layer = nn.Linear(20, 10)
        x = FloatTensor(4, 20).normal_()
        out = layer.forward(x)
        assert out.shape == (4, 10)

    def test_parameters(self):
        from torch9.luatorch import nn

        layer = nn.Linear(20, 10)
        params, grads = layer.parameters()
        assert len(params) == 2  # weight + bias
        assert params[0].shape == (10, 20)
        assert params[1].shape == (10,)
        assert grads[0].shape == (10, 20)
        assert grads[1].shape == (10,)

    def test_no_bias(self):
        from torch9.luatorch import nn, FloatTensor

        layer = nn.Linear(5, 3, bias=False)
        params, grads = layer.parameters()
        assert len(params) == 1
        x = FloatTensor(2, 5).normal_()
        out = layer.forward(x)
        assert out.shape == (2, 3)

    def test_backward(self):
        from torch9.luatorch import nn, FloatTensor

        layer = nn.Linear(4, 3)
        x = FloatTensor(2, 4).normal_()
        out = layer.forward(x)
        grad_out = torch.ones_like(out)
        layer.zeroGradParameters()
        grad_in = layer.backward(x, grad_out)
        assert grad_in.shape == x.shape
        _, grads = layer.parameters()
        assert grads[0].abs().sum() > 0  # gradWeight non-zero

    def test_zero_grad(self):
        from torch9.luatorch import nn, FloatTensor

        layer = nn.Linear(4, 3)
        x = FloatTensor(2, 4).normal_()
        layer.forward(x)
        layer.backward(x, torch.ones(2, 3))
        layer.zeroGradParameters()
        _, grads = layer.parameters()
        assert grads[0].abs().sum() == 0

    def test_update_parameters(self):
        from torch9.luatorch import nn, FloatTensor

        layer = nn.Linear(4, 3)
        x = FloatTensor(2, 4).normal_()
        original_weight = layer.weight.clone()
        layer.forward(x)
        layer.backward(x, torch.ones(2, 3))
        layer.updateParameters(0.01)
        assert not torch.allclose(layer.weight, original_weight)

    def test_repr(self):
        from torch9.luatorch import nn

        layer = nn.Linear(10, 5)
        assert "10 -> 5" in repr(layer)


# =============================================================================
# nn.Sequential
# =============================================================================


class TestSequential:
    """Tests for Sequential container."""

    def test_forward(self):
        from torch9.luatorch import nn, FloatTensor

        model = nn.Sequential()
        model.add(nn.Linear(10, 5))
        model.add(nn.ReLU())
        model.add(nn.Linear(5, 2))
        x = FloatTensor(3, 10).normal_()
        out = model.forward(x)
        assert out.shape == (3, 2)

    def test_backward(self):
        from torch9.luatorch import nn, FloatTensor

        model = nn.Sequential()
        model.add(nn.Linear(10, 5))
        model.add(nn.ReLU())
        model.add(nn.Linear(5, 2))
        x = FloatTensor(3, 10).normal_()
        out = model.forward(x)
        model.zeroGradParameters()
        grad_in = model.backward(x, torch.ones_like(out))
        assert grad_in.shape == x.shape

    def test_size(self):
        from torch9.luatorch import nn

        model = nn.Sequential()
        model.add(nn.Linear(10, 5))
        model.add(nn.ReLU())
        assert model.size() == 2

    def test_get(self):
        from torch9.luatorch import nn

        model = nn.Sequential()
        linear = nn.Linear(10, 5)
        model.add(linear)
        assert model.get(0) is linear

    def test_insert(self):
        from torch9.luatorch import nn

        model = nn.Sequential()
        model.add(nn.Linear(10, 5))
        model.add(nn.Linear(5, 2))
        relu = nn.ReLU()
        model.insert(relu, 1)
        assert model.size() == 3
        assert model.get(1) is relu

    def test_remove(self):
        from torch9.luatorch import nn

        model = nn.Sequential()
        model.add(nn.Linear(10, 5))
        relu = nn.ReLU()
        model.add(relu)
        removed = model.remove(1)
        assert removed is relu
        assert model.size() == 1

    def test_parameters_recursive(self):
        from torch9.luatorch import nn

        model = nn.Sequential()
        model.add(nn.Linear(10, 5))
        model.add(nn.ReLU())
        model.add(nn.Linear(5, 2))
        params, grads = model.parameters()
        assert len(params) == 4  # 2 layers * (weight + bias)

    def test_training_propagation(self):
        from torch9.luatorch import nn

        model = nn.Sequential()
        model.add(nn.Dropout(0.5))
        model.evaluate()
        assert model.modules[0].train_mode is False
        model.training()
        assert model.modules[0].train_mode is True

    def test_repr(self):
        from torch9.luatorch import nn

        model = nn.Sequential()
        model.add(nn.Linear(10, 5))
        model.add(nn.ReLU())
        r = repr(model)
        assert "nn.Sequential" in r
        assert "nn.Linear" in r
        assert "nn.ReLU" in r

    def test_chaining(self):
        from torch9.luatorch import nn

        model = nn.Sequential()
        result = model.add(nn.Linear(10, 5))
        assert result is model  # add returns self


# =============================================================================
# nn.ConcatTable
# =============================================================================


class TestConcatTable:
    """Tests for ConcatTable container."""

    def test_forward(self):
        from torch9.luatorch import nn, FloatTensor

        ct = nn.ConcatTable()
        ct.add(nn.Linear(10, 5))
        ct.add(nn.Linear(10, 3))
        x = FloatTensor(2, 10).normal_()
        outputs = ct.forward(x)
        assert len(outputs) == 2
        assert outputs[0].shape == (2, 5)
        assert outputs[1].shape == (2, 3)


# =============================================================================
# nn.Parallel
# =============================================================================


class TestParallel:
    """Tests for Parallel container."""

    def test_forward(self):
        from torch9.luatorch import nn, FloatTensor

        par = nn.Parallel(input_dim=1, output_dim=1)
        par.add(nn.Linear(5, 3))
        par.add(nn.Linear(5, 3))
        x = FloatTensor(2, 10).normal_()  # split 10 into 2x5
        out = par.forward(x)
        assert out.shape == (2, 6)  # cat 3+3=6 along dim=1


# =============================================================================
# Activation functions
# =============================================================================


class TestActivations:
    """Tests for activation function modules."""

    def test_relu_forward(self):
        from torch9.luatorch import nn

        relu = nn.ReLU()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = relu.forward(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        assert torch.allclose(out, expected)

    def test_relu_backward(self):
        from torch9.luatorch import nn

        relu = nn.ReLU()
        x = torch.tensor([-1.0, 0.0, 1.0])
        relu.forward(x)
        grad = relu.backward(x, torch.ones(3))
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(grad, expected)

    def test_tanh(self):
        from torch9.luatorch import nn

        tanh = nn.Tanh()
        x = torch.tensor([0.0, 1.0, -1.0])
        out = tanh.forward(x)
        assert torch.allclose(out, torch.tanh(x))

    def test_tanh_backward(self):
        from torch9.luatorch import nn

        tanh = nn.Tanh()
        x = torch.tensor([0.5, -0.5])
        out = tanh.forward(x)
        grad = tanh.backward(x, torch.ones(2))
        expected = 1.0 - out ** 2
        assert torch.allclose(grad, expected)

    def test_sigmoid(self):
        from torch9.luatorch import nn

        sig = nn.Sigmoid()
        x = torch.tensor([0.0, 2.0, -2.0])
        out = sig.forward(x)
        assert torch.allclose(out, torch.sigmoid(x))

    def test_sigmoid_backward(self):
        from torch9.luatorch import nn

        sig = nn.Sigmoid()
        x = torch.tensor([0.0])
        out = sig.forward(x)
        grad = sig.backward(x, torch.ones(1))
        expected = out * (1 - out)
        assert torch.allclose(grad, expected)

    def test_log_softmax(self):
        from torch9.luatorch import nn

        lsm = nn.LogSoftMax()
        x = torch.randn(2, 5)
        out = lsm.forward(x)
        expected = torch.log_softmax(x, dim=1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_softmax(self):
        from torch9.luatorch import nn

        sm = nn.SoftMax()
        x = torch.randn(2, 5)
        out = sm.forward(x)
        expected = torch.softmax(x, dim=1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_leaky_relu(self):
        from torch9.luatorch import nn

        lrelu = nn.LeakyReLU(0.1)
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = lrelu.forward(x)
        expected = torch.tensor([-0.2, -0.1, 0.0, 1.0, 2.0])
        assert torch.allclose(out, expected)

    def test_elu(self):
        from torch9.luatorch import nn

        elu = nn.ELU(1.0)
        x = torch.tensor([1.0, 0.0, -1.0])
        out = elu.forward(x)
        assert out[0] == 1.0
        assert out[1] == 0.0
        assert out[2] < 0  # alpha * (exp(-1) - 1)


# =============================================================================
# nn.SpatialConvolution
# =============================================================================


class TestSpatialConvolution:
    """Tests for 2D convolution layer."""

    def test_output_shape(self):
        from torch9.luatorch import nn, FloatTensor

        conv = nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1)
        x = FloatTensor(1, 3, 8, 8).normal_()
        out = conv.forward(x)
        assert out.shape == (1, 16, 8, 8)

    def test_no_padding_shape(self):
        from torch9.luatorch import nn, FloatTensor

        conv = nn.SpatialConvolution(3, 16, 3, 3)
        x = FloatTensor(1, 3, 8, 8).normal_()
        out = conv.forward(x)
        assert out.shape == (1, 16, 6, 6)

    def test_parameters(self):
        from torch9.luatorch import nn

        conv = nn.SpatialConvolution(3, 16, 3, 3)
        params, grads = conv.parameters()
        assert params[0].shape == (16, 3, 3, 3)  # weight
        assert params[1].shape == (16,)  # bias

    def test_backward(self):
        from torch9.luatorch import nn, FloatTensor

        conv = nn.SpatialConvolution(3, 8, 3, 3, 1, 1, 1, 1)
        x = FloatTensor(2, 3, 8, 8).normal_()
        out = conv.forward(x)
        conv.zeroGradParameters()
        grad_in = conv.backward(x, torch.ones_like(out))
        assert grad_in.shape == x.shape

    def test_repr(self):
        from torch9.luatorch import nn

        conv = nn.SpatialConvolution(3, 16, 5, 5, 2, 2, 1, 1)
        r = repr(conv)
        assert "SpatialConvolution" in r
        assert "3 -> 16" in r


# =============================================================================
# nn.SpatialBatchNormalization
# =============================================================================


class TestSpatialBatchNorm:
    """Tests for spatial batch normalization."""

    def test_output_shape(self):
        from torch9.luatorch import nn, FloatTensor

        bn = nn.SpatialBatchNormalization(16)
        x = FloatTensor(4, 16, 8, 8).normal_()
        out = bn.forward(x)
        assert out.shape == x.shape

    def test_running_stats_updated(self):
        from torch9.luatorch import nn, FloatTensor

        bn = nn.SpatialBatchNormalization(8)
        x = FloatTensor(4, 8, 4, 4).normal_()
        bn.training()
        bn.forward(x)
        assert not torch.allclose(bn.running_mean, torch.zeros(8), atol=1e-6)

    def test_eval_mode_uses_running_stats(self):
        from torch9.luatorch import nn, FloatTensor

        bn = nn.SpatialBatchNormalization(4)
        # Train to build running stats
        for _ in range(10):
            x = FloatTensor(8, 4, 4, 4).normal_()
            bn.forward(x)
        running_mean = bn.running_mean.clone()
        # Switch to eval
        bn.evaluate()
        x = FloatTensor(2, 4, 4, 4).normal_()
        bn.forward(x)
        # running_mean should not change
        assert torch.allclose(bn.running_mean, running_mean)

    def test_parameters(self):
        from torch9.luatorch import nn

        bn = nn.SpatialBatchNormalization(8, affine=True)
        params, grads = bn.parameters()
        assert len(params) == 2  # weight + bias

    def test_no_affine(self):
        from torch9.luatorch import nn

        bn = nn.SpatialBatchNormalization(8, affine=False)
        assert bn.parameters() is None


# =============================================================================
# nn.Dropout
# =============================================================================


class TestDropout:
    """Tests for dropout regularization."""

    def test_training_drops(self):
        from torch9.luatorch import nn, FloatTensor

        drop = nn.Dropout(0.5)
        drop.training()
        x = FloatTensor(1000).fill_(1.0)
        out = drop.forward(x)
        zeros = (out == 0).sum().item()
        assert zeros > 100  # Some elements dropped

    def test_eval_no_drop(self):
        from torch9.luatorch import nn, FloatTensor

        drop = nn.Dropout(0.5)
        drop.evaluate()
        x = FloatTensor(100).fill_(1.0)
        out = drop.forward(x)
        assert torch.allclose(out, x)

    def test_backward(self):
        from torch9.luatorch import nn, FloatTensor

        drop = nn.Dropout(0.5)
        drop.training()
        x = FloatTensor(100).normal_()
        drop.forward(x)
        grad = drop.backward(x, torch.ones(100))
        # Grad should have same mask pattern as output
        assert grad.shape == x.shape


# =============================================================================
# nn.View and nn.Identity
# =============================================================================


class TestReshapeModules:
    """Tests for View and Identity."""

    def test_view(self):
        from torch9.luatorch import nn, FloatTensor

        view = nn.View(2, 6)
        x = FloatTensor(12).normal_()
        out = view.forward(x)
        assert out.shape == (2, 6)

    def test_view_backward(self):
        from torch9.luatorch import nn, FloatTensor

        view = nn.View(2, 6)
        x = FloatTensor(12).normal_()
        view.forward(x)
        grad = view.backward(x, torch.ones(2, 6))
        assert grad.shape == x.shape

    def test_identity(self):
        from torch9.luatorch import nn, FloatTensor

        identity = nn.Identity()
        x = FloatTensor(3, 4).normal_()
        out = identity.forward(x)
        assert torch.equal(out, x)

    def test_identity_backward(self):
        from torch9.luatorch import nn

        identity = nn.Identity()
        x = torch.randn(3, 4)
        identity.forward(x)
        grad_out = torch.randn(3, 4)
        grad = identity.backward(x, grad_out)
        assert torch.equal(grad, grad_out)


# =============================================================================
# Criterion (Loss Functions)
# =============================================================================


class TestMSECriterion:
    """Tests for MSE loss."""

    def test_forward(self):
        from torch9.luatorch import nn

        crit = nn.MSECriterion()
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = crit.forward(pred, target)
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_forward_nonzero(self):
        from torch9.luatorch import nn

        crit = nn.MSECriterion()
        pred = torch.tensor([1.0, 0.0])
        target = torch.tensor([0.0, 0.0])
        loss = crit.forward(pred, target)
        assert loss == pytest.approx(0.5, abs=1e-6)

    def test_backward(self):
        from torch9.luatorch import nn

        crit = nn.MSECriterion()
        pred = torch.tensor([2.0, 3.0])
        target = torch.tensor([1.0, 1.0])
        crit.forward(pred, target)
        grad = crit.backward(pred, target)
        assert grad.shape == pred.shape


class TestClassNLLCriterion:
    """Tests for negative log-likelihood loss."""

    def test_forward(self):
        from torch9.luatorch import nn

        crit = nn.ClassNLLCriterion()
        log_probs = torch.log_softmax(torch.randn(4, 5), dim=1)
        targets = torch.tensor([0, 1, 2, 3])
        loss = crit.forward(log_probs, targets)
        assert loss > 0

    def test_backward_shape(self):
        from torch9.luatorch import nn

        crit = nn.ClassNLLCriterion()
        log_probs = torch.log_softmax(torch.randn(4, 5), dim=1)
        targets = torch.tensor([0, 1, 2, 3])
        crit.forward(log_probs, targets)
        grad = crit.backward(log_probs, targets)
        assert grad.shape == log_probs.shape


class TestCrossEntropyCriterion:
    """Tests for cross-entropy loss."""

    def test_forward(self):
        from torch9.luatorch import nn

        crit = nn.CrossEntropyCriterion()
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        loss = crit.forward(logits, targets)
        assert loss > 0

    def test_backward_shape(self):
        from torch9.luatorch import nn

        crit = nn.CrossEntropyCriterion()
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        crit.forward(logits, targets)
        grad = crit.backward(logits, targets)
        assert grad.shape == logits.shape


class TestBCECriterion:
    """Tests for binary cross-entropy loss."""

    def test_forward(self):
        from torch9.luatorch import nn

        crit = nn.BCECriterion()
        pred = torch.tensor([0.5, 0.5])
        target = torch.tensor([1.0, 0.0])
        loss = crit.forward(pred, target)
        assert loss > 0

    def test_backward_shape(self):
        from torch9.luatorch import nn

        crit = nn.BCECriterion()
        pred = torch.sigmoid(torch.randn(4))
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        crit.forward(pred, target)
        grad = crit.backward(pred, target)
        assert grad.shape == pred.shape


class TestAbsCriterion:
    """Tests for L1 loss."""

    def test_forward(self):
        from torch9.luatorch import nn

        crit = nn.AbsCriterion()
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([0.0, 0.0])
        loss = crit.forward(pred, target)
        assert loss == pytest.approx(1.5, abs=1e-6)


class TestSmoothL1Criterion:
    """Tests for Smooth L1 loss."""

    def test_forward(self):
        from torch9.luatorch import nn

        crit = nn.SmoothL1Criterion()
        pred = torch.tensor([0.5, 2.0])
        target = torch.tensor([0.0, 0.0])
        loss = crit.forward(pred, target)
        assert loss > 0


# =============================================================================
# Functional Optimizers
# =============================================================================


class TestOptimSGD:
    """Tests for functional SGD optimizer."""

    def test_basic_step(self):
        from torch9.luatorch import optim

        x = torch.tensor([5.0, 5.0])
        state = {}

        def opfunc(x):
            return x.sum().item(), torch.ones_like(x)

        x, losses = optim.sgd(opfunc, x, {"learningRate": 0.1}, state)
        assert torch.allclose(x, torch.tensor([4.9, 4.9]))
        assert losses[0] == 10.0

    def test_momentum(self):
        from torch9.luatorch import optim

        x = torch.tensor([5.0])
        state = {}
        # dampening=0 so momentum buffer accumulates (Torch7 default is mom)
        config = {"learningRate": 0.1, "momentum": 0.9, "dampening": 0}

        def opfunc(x):
            return x.item(), torch.ones(1)

        optim.sgd(opfunc, x, config, state)
        first = x.item()
        optim.sgd(opfunc, x, config, state)
        second = x.item()
        optim.sgd(opfunc, x, config, state)
        third = x.item()
        # With momentum and zero dampening, step sizes increase
        step1 = first - second
        step2 = second - third
        assert step2 > step1  # Momentum accelerates

    def test_weight_decay(self):
        from torch9.luatorch import optim

        x = torch.tensor([2.0])
        state = {}
        config = {"learningRate": 0.1, "weightDecay": 0.01}

        def opfunc(x):
            return 0.0, torch.zeros(1)

        optim.sgd(opfunc, x, config, state)
        # Even with zero grad, weight decay pulls x toward 0
        assert x.item() < 2.0

    def test_eval_counter(self):
        from torch9.luatorch import optim

        x = torch.tensor([1.0])
        state = {}

        def opfunc(x):
            return 0.0, torch.ones(1)

        optim.sgd(opfunc, x, {}, state)
        assert state["evalCounter"] == 1
        optim.sgd(opfunc, x, {}, state)
        assert state["evalCounter"] == 2


class TestOptimAdam:
    """Tests for functional Adam optimizer."""

    def test_basic_step(self):
        from torch9.luatorch import optim

        x = torch.tensor([5.0, 5.0])
        state = {}

        def opfunc(x):
            return x.sum().item(), torch.ones_like(x)

        x, losses = optim.adam(opfunc, x, {"learningRate": 0.01}, state)
        assert x[0].item() < 5.0  # Moved toward 0
        assert "m" in state
        assert "v" in state

    def test_convergence(self):
        from torch9.luatorch import optim

        x = torch.tensor([10.0])
        state = {}
        config = {"learningRate": 0.1}

        def opfunc(x):
            loss = (x ** 2).sum().item()
            grad = 2 * x
            return loss, grad

        for _ in range(500):
            optim.adam(opfunc, x, config, state)
        assert abs(x.item()) < 1.0  # Should converge toward 0


class TestOptimAdagrad:
    """Tests for functional Adagrad optimizer."""

    def test_basic_step(self):
        from torch9.luatorch import optim

        x = torch.tensor([5.0])
        state = {}

        def opfunc(x):
            return x.item(), torch.ones(1)

        optim.adagrad(opfunc, x, {"learningRate": 0.1}, state)
        assert x.item() < 5.0
        assert "sum_sq" in state


class TestOptimRMSProp:
    """Tests for functional RMSProp optimizer."""

    def test_basic_step(self):
        from torch9.luatorch import optim

        x = torch.tensor([5.0])
        state = {}

        def opfunc(x):
            return x.item(), torch.ones(1)

        optim.rmsprop(opfunc, x, {"learningRate": 0.01}, state)
        assert x.item() < 5.0
        assert "v" in state


# =============================================================================
# Serialization
# =============================================================================


class TestSerialization:
    """Tests for save/load."""

    def test_save_load_tensor(self):
        from torch9.luatorch import save, load

        t = torch.randn(3, 4)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save(path, t)
            loaded = load(path)
            assert torch.allclose(t, loaded)
        finally:
            os.unlink(path)

    def test_save_load_dict(self):
        from torch9.luatorch import save, load

        data = {"weight": torch.randn(5, 3), "bias": torch.randn(3)}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save(path, data)
            loaded = load(path)
            assert torch.allclose(data["weight"], loaded["weight"])
            assert torch.allclose(data["bias"], loaded["bias"])
        finally:
            os.unlink(path)


# =============================================================================
# Timer
# =============================================================================


class TestTimer:
    """Tests for tic/toc timer."""

    def test_timer_basic(self):
        from torch9.luatorch import Timer

        t = Timer()
        t.tic()
        elapsed = t.toc()
        assert elapsed >= 0

    def test_global_tic_toc(self):
        from torch9.luatorch import tic, toc

        tic()
        elapsed = toc()
        assert elapsed >= 0

    def test_toc_without_tic_raises(self):
        from torch9.luatorch import Timer

        t = Timer()
        with pytest.raises(RuntimeError, match="tic"):
            t.toc()

    def test_timer_repr(self):
        from torch9.luatorch import Timer

        t = Timer()
        assert "stopped" in repr(t)
        t.tic()
        assert "running" in repr(t)


# =============================================================================
# Paths utility
# =============================================================================


class TestPaths:
    """Tests for paths utility namespace."""

    def test_home(self):
        from torch9.luatorch import paths

        h = paths.home()
        assert os.path.isdir(h)

    def test_concat(self):
        from torch9.luatorch import paths

        p = paths.concat("/tmp", "foo", "bar.txt")
        assert p == os.path.join("/tmp", "foo", "bar.txt")

    def test_filep(self):
        from torch9.luatorch import paths

        assert paths.filep(__file__)
        assert not paths.filep("/nonexistent_file_xyz")

    def test_dirp(self):
        from torch9.luatorch import paths

        assert paths.dirp(os.path.dirname(__file__))

    def test_basename(self):
        from torch9.luatorch import paths

        assert paths.basename("/tmp/foo/bar.txt") == "bar.txt"

    def test_dirname(self):
        from torch9.luatorch import paths

        assert paths.dirname("/tmp/foo/bar.txt") == "/tmp/foo"

    def test_extname(self):
        from torch9.luatorch import paths

        assert paths.extname("file.py") == ".py"
        assert paths.extname("noext") == ""

    def test_mkdir_and_rmall(self):
        from torch9.luatorch import paths

        dirpath = os.path.join(tempfile.gettempdir(), "torch9_test_paths_dir")
        try:
            paths.mkdir(dirpath)
            assert paths.dirp(dirpath)
        finally:
            if os.path.isdir(dirpath):
                paths.rmall(dirpath)


# =============================================================================
# End-to-end integration
# =============================================================================


class TestEndToEnd:
    """Integration tests for a full training-style workflow."""

    def test_full_forward_backward(self):
        """Build model, run forward, compute loss, run backward, update."""
        from torch9.luatorch import nn, FloatTensor

        model = nn.Sequential()
        model.add(nn.Linear(8, 16))
        model.add(nn.ReLU())
        model.add(nn.Dropout(0.1))
        model.add(nn.Linear(16, 4))
        model.add(nn.LogSoftMax())

        criterion = nn.ClassNLLCriterion()

        x = FloatTensor(4, 8).normal_()
        targets = torch.tensor([0, 1, 2, 3])

        # Forward
        output = model.forward(x)
        loss = criterion.forward(output, targets)
        assert loss > 0
        assert output.shape == (4, 4)

        # Backward
        grad_output = criterion.backward(output, targets)
        model.zeroGradParameters()
        model.backward(x, grad_output)

        # Update
        model.updateParameters(0.01)

    def test_optim_training_loop(self):
        """Use functional optimizer for a few steps."""
        from torch9.luatorch import nn, optim, FloatTensor

        model = nn.Sequential()
        model.add(nn.Linear(4, 8))
        model.add(nn.ReLU())
        model.add(nn.Linear(8, 2))

        criterion = nn.MSECriterion()

        x = FloatTensor(8, 4).normal_()
        target = FloatTensor(8, 2).fill_(0.0)

        # Flatten parameters for optimizer
        params, grad_params = model.parameters()
        flat_p = torch.cat([p.flatten() for p in params])
        state = {}

        losses = []
        for _ in range(5):
            def opfunc(flat_x):
                # Unflatten into params
                offset = 0
                for p in params:
                    numel = p.numel()
                    p.data.copy_(flat_x[offset:offset + numel].view_as(p))
                    offset += numel
                model.zeroGradParameters()
                out = model.forward(x)
                loss = criterion.forward(out, target)
                grad = criterion.backward(out, target)
                model.backward(x, grad)
                flat_g = torch.cat([g.flatten() for g in grad_params])
                return loss, flat_g

            flat_p, fx = optim.sgd(
                opfunc, flat_p, {"learningRate": 0.01}, state
            )
            losses.append(fx[0])

        # Loss should decrease
        assert losses[-1] <= losses[0]

    def test_conv_model(self):
        """Test a convolution-based model."""
        from torch9.luatorch import nn, FloatTensor

        model = nn.Sequential()
        model.add(nn.SpatialConvolution(1, 4, 3, 3, 1, 1, 1, 1))
        model.add(nn.SpatialBatchNormalization(4))
        model.add(nn.ReLU())

        x = FloatTensor(2, 1, 8, 8).normal_()
        out = model.forward(x)
        assert out.shape == (2, 4, 8, 8)

        model.zeroGradParameters()
        grad = model.backward(x, torch.ones_like(out))
        assert grad.shape == x.shape
