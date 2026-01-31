# ReservoirLua: Implementation Guide

## Overview

This document describes the implementation of reservoirlua, a Lua port of the reservoirpy Python library for the Torch7 framework. The implementation provides Reservoir Computing (Echo State Networks) functionality compatible with the existing rnn library.

## Key Features Implemented

### 1. Echo State Network (ESN) Module

The core `nn.Reservoir` module implements a complete Echo State Network with the following features:

- **Fixed Reservoir Weights**: Randomly initialized and never trained
- **Trainable Readout**: Only the output layer (Wout, bias) is trained
- **Spectral Radius Control**: Power iteration method to scale reservoir dynamics
- **Sparse Connectivity**: Configurable sparsity for both reservoir and input weights
- **Leak Rate**: Leaky integrator neurons for temporal processing
- **Batch Processing**: Supports both single sample and batch mode

### 2. Architecture

```
Input (x_t) ──→ Win ──┐
                      │
                      ↓
              ┌──→ Reservoir State (h_t) ──┐
              │       ↑                     │
              │       │                     │
              └───── W (fixed)              │
                                            │
                                            ↓
                                    [x_t ; h_t] ──→ Wout ──→ Output (y_t)
                                                     (trainable)
```

### 3. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| spectralRadius | 0.9 | Controls reservoir dynamics (< 1 for stability) |
| inputScaling | 1.0 | Scaling of input weights |
| leakRate | 1.0 | Leak rate for leaky integrator (0-1) |
| connectivity | 0.1 | Sparsity of reservoir weights (0-1) |
| inputConnectivity | 0.1 | Sparsity of input weights (0-1) |
| seed | nil | Random seed for reproducibility |

## Comparison with reservoirpy

### Similarities

Both implementations provide:
1. Echo State Networks with fixed reservoir weights
2. Spectral radius control
3. Sparse connectivity
4. Leak rate for temporal dynamics
5. Fast training (only output layer)
6. Similar API design

### Differences

| Feature | reservoirlua | reservoirpy |
|---------|--------------|-------------|
| Language | Lua | Python |
| Framework | Torch7 | NumPy/SciPy |
| Backend | C/CUDA via Torch | NumPy/BLAS |
| API Style | Object-oriented (nn.Module) | Object-oriented (Node) |
| GPU Support | Native via cutorch | Via CuPy |
| Integration | rnn library | Standalone |

### Code Comparison

**reservoirpy (Python):**
```python
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(
    units=100,
    sr=0.9,
    lr=0.3,
    input_scaling=1.0,
    rc_connectivity=0.1,
    input_connectivity=0.2
)

readout = Ridge(output_dim=1)
esn = reservoir >> readout

# Train
esn.fit(X_train, Y_train)

# Predict
Y_pred = esn.run(X_test)
```

**reservoirlua (Lua):**
```lua
require 'rnn'

local options = {
   spectralRadius = 0.9,
   leakRate = 0.3,
   inputScaling = 1.0,
   connectivity = 0.1,
   inputConnectivity = 0.2
}

local reservoir = nn.Reservoir(inputSize, 100, outputSize, options)

-- Train
for t = 1, seqlen do
   local output = reservoir:forward(X_train[t])
   local gradOutput = criterion:backward(output, Y_train[t])
   reservoir:backward(X_train[t], gradOutput)
end
reservoir:updateParameters(lr)

-- Predict
reservoir:forget()
for t = 1, testlen do
   Y_pred[t] = reservoir:forward(X_test[t])
end
```

## Implementation Details

### Weight Initialization

1. **Input Weights (Win)**: Uniformly initialized in [-inputScaling, +inputScaling]
2. **Reservoir Weights (W)**: Uniformly initialized in [-0.5, +0.5], then scaled to spectral radius
3. **Output Weights (Wout)**: Uniformly initialized in [-0.1, +0.1]

### Spectral Radius Scaling

Uses power iteration method (100 iterations) to approximate the largest eigenvalue:
```lua
function Reservoir:scaleToSpectralRadius()
   local v = torch.randn(self.reservoirSize, 1)
   v:div(v:norm())
   
   for i = 1, 100 do
      v = self.W * v
      v:div(v:norm())
   end
   
   local lambda = (self.W * v):norm()
   if lambda > 0 then
      self.W:mul(self.spectralRadius / lambda)
   end
end
```

### State Update Equation

ESN update with leak rate:
```
h(t) = (1 - α) * h(t-1) + α * tanh(Win * x(t) + W * h(t-1))
```

where α is the leak rate.

### Training

Only the readout layer (Wout, bias) is trained:
```lua
function Reservoir:accGradParameters(input, gradOutput, scale)
   local extendedState = torch.cat(input, self.state)
   self.gradWout:addr(scale, gradOutput, extendedState)
   self.gradBias:add(scale, gradOutput)
end
```

## Usage Patterns

### 1. Time Series Prediction
```lua
local reservoir = nn.Reservoir(inputSize, reservoirSize, outputSize, options)
local criterion = nn.MSECriterion()

for iter = 1, numIterations do
   reservoir:forget()
   reservoir:zeroGradParameters()
   
   for t = 1, seqlen do
      local output = reservoir:forward(input[t])
      local loss = criterion:forward(output, target[t])
      local gradOutput = criterion:backward(output, target[t])
      reservoir:backward(input[t], gradOutput)
   end
   
   reservoir:updateParameters(learningRate)
end
```

### 2. Integration with Sequencer
```lua
local reservoir = nn.Reservoir(inputSize, reservoirSize, outputSize)
local model = nn.Sequential()
   :add(nn.Sequencer(reservoir))
   :add(nn.SelectTable(-1))  -- Select last timestep
   :add(nn.Linear(outputSize, numClasses))
```

### 3. Multi-step Prediction
```lua
-- Train
reservoir:forget()
for t = 1, trainlen do
   reservoir:forward(trainData[t])
end

-- Predict
local predictions = {}
local lastInput = trainData[trainlen]
for t = 1, horizon do
   local prediction = reservoir:forward(lastInput)
   predictions[t] = prediction
   lastInput = prediction  -- Feed prediction back as input
end
```

## Performance Considerations

1. **Sparsity**: Lower connectivity values reduce computation time
2. **Reservoir Size**: Larger reservoirs capture more dynamics but are slower
3. **Batch Processing**: Use batches when possible for efficiency
4. **GPU**: Use cutorch for large-scale problems

## Testing

The implementation includes comprehensive tests in `test/test-reservoir.lua`:
- Basic initialization and forward pass
- Custom options
- Gradient computation
- Sequence processing
- Parameter updates
- Deterministic behavior with seeds

## Future Enhancements

Potential additions to match more reservoirpy features:
1. Additional activation functions (identity, softmax, etc.)
2. Different weight initialization schemes
3. Intrinsic Plasticity for automatic parameter tuning
4. Deep reservoir architectures
5. Bidirectional reservoirs
6. Online learning methods

## References

1. reservoirpy: https://github.com/reservoirpy/reservoirpy
2. Lukoševičius, M. (2012). "A Practical Guide to Applying Echo State Networks"
3. Jaeger, H. (2001). "The echo state approach to analysing and training recurrent neural networks"
4. Torch RNN library: https://github.com/torch/rnn

## License

BSD License (same as the rnn library)
