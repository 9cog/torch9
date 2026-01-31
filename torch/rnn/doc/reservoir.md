# ReservoirLua - Reservoir Computing for Torch

A Lua implementation of Reservoir Computing (Echo State Networks) for Torch7, inspired by the [reservoirpy](https://github.com/reservoirpy/reservoirpy) Python library.

## Overview

Reservoir Computing is a machine learning paradigm particularly suited for processing time-series and sequential data. The key idea is to use a fixed, randomly initialized recurrent neural network (the "reservoir") and only train a simple output layer (the "readout"). This makes training very fast while still capturing complex temporal dynamics.

## Features

- **Echo State Networks (ESN)** - The most popular reservoir computing architecture
- **Configurable Parameters** - Control spectral radius, input scaling, leak rate, and connectivity
- **Sparse Connectivity** - Efficient sparse reservoir and input weights
- **Fast Training** - Only the readout layer is trained (linear regression)
- **Compatible** - Works seamlessly with the existing rnn library modules

## Installation

```bash
luarocks install reservoirlua
```

Or if using from the rnn library source:
```lua
require 'rnn'
-- Reservoir module is automatically loaded
```

## Quick Start

```lua
require 'rnn'

-- Create a reservoir with custom options
local options = {
   spectralRadius = 0.9,     -- Controls dynamics (< 1 for stability)
   inputScaling = 1.0,       -- Scaling of input weights
   leakRate = 0.3,           -- Leak rate for leaky integrator (0-1)
   connectivity = 0.1,       -- Sparsity of reservoir weights (0-1)
   inputConnectivity = 0.2,  -- Sparsity of input weights (0-1)
   seed = 12345              -- Random seed for reproducibility
}

local reservoir = nn.Reservoir(inputSize, reservoirSize, outputSize, options)

-- Process a time series
reservoir:forget()  -- Reset state
for t = 1, seqlen do
   local output = reservoir:forward(input[t])
   -- ... training or prediction ...
end
```

## API Reference

### nn.Reservoir(inputSize, reservoirSize, outputSize, [options])

Creates a new Reservoir (Echo State Network) module.

**Parameters:**
- `inputSize` (number): Dimension of input vectors
- `reservoirSize` (number): Number of neurons in the reservoir
- `outputSize` (number): Dimension of output vectors
- `options` (table, optional): Configuration options
  - `spectralRadius` (number, default=0.9): Spectral radius of reservoir weights
  - `inputScaling` (number, default=1.0): Scaling factor for input weights
  - `leakRate` (number, default=1.0): Leak rate (alpha) for leaky integrator
  - `connectivity` (number, default=0.1): Sparsity of reservoir weights (0-1)
  - `inputConnectivity` (number, default=0.1): Sparsity of input weights (0-1)
  - `seed` (number, optional): Random seed for weight initialization

**Returns:** Reservoir module

### Methods

- `reservoir:forward(input)` - Process one time step
- `reservoir:backward(input, gradOutput)` - Compute gradients
- `reservoir:forget()` - Reset reservoir state to zero
- `reservoir:updateParameters(learningRate)` - Update readout weights
- `reservoir:zeroGradParameters()` - Zero the gradients

## Examples

### Time Series Prediction

See [examples/simple-reservoir-network.lua](examples/simple-reservoir-network.lua) for a complete example of time series prediction using reservoir computing.

### Typical Use Cases

1. **Time Series Forecasting** - Stock prices, weather, sensor data
2. **Signal Processing** - Audio, speech, physiological signals
3. **Pattern Recognition** - Sequential pattern detection
4. **Control Systems** - Dynamical system modeling

## Key Concepts

### Spectral Radius
The spectral radius controls the dynamics of the reservoir:
- Values < 1.0: Stable, fading memory
- Values ≈ 1.0: Critical regime, edge of chaos (often best)
- Values > 1.0: Unstable, chaotic dynamics

### Leak Rate
The leak rate (alpha) controls temporal integration:
- alpha = 1.0: No leakage, standard ESN
- alpha < 1.0: Leaky integrator, slower dynamics
- Smaller values = longer memory

### Connectivity
Sparsity of connections:
- Lower values = more sparse, faster computation
- Higher values = more connections, potentially better performance
- Typical range: 0.01 to 0.3

## Comparison with reservoirpy

This implementation provides similar functionality to the Python reservoirpy library:

| Feature | reservoirlua | reservoirpy |
|---------|--------------|-------------|
| Echo State Networks | ✓ | ✓ |
| Spectral Radius Control | ✓ | ✓ |
| Sparse Connectivity | ✓ | ✓ |
| Leak Rate | ✓ | ✓ |
| Fast Training | ✓ | ✓ |
| Online Learning | ✓ | ✓ |

## References

1. Lukoševičius, M. (2012). "A Practical Guide to Applying Echo State Networks" - Neural Networks: Tricks of the Trade
2. Jaeger, H. (2001). "The echo state approach to analysing and training recurrent neural networks"
3. reservoirpy: https://github.com/reservoirpy/reservoirpy

## License

BSD License (same as the rnn library)

## Contributing

Contributions are welcome! Please submit issues and pull requests to the main rnn repository.
