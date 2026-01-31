require 'rnn'

local ReservoirTest = {}
local precision = 1e-5
local mytester

function ReservoirTest.test_basic()
   -- Test basic initialization and forward pass
   local inputSize = 5
   local reservoirSize = 20
   local outputSize = 3
   local batchSize = 4
   
   local reservoir = nn.Reservoir(inputSize, reservoirSize, outputSize)
   mytester:asserteq(reservoir.inputSize, inputSize, 'inputSize mismatch')
   mytester:asserteq(reservoir.reservoirSize, reservoirSize, 'reservoirSize mismatch')
   mytester:asserteq(reservoir.outputSize, outputSize, 'outputSize mismatch')
   
   -- Test single sample forward
   local input = torch.randn(inputSize)
   local output = reservoir:forward(input)
   mytester:asserteq(output:dim(), 1, 'output should be 1D for single sample')
   mytester:asserteq(output:size(1), outputSize, 'output size mismatch')
   
   -- Test batch forward
   reservoir:forget()
   local batchInput = torch.randn(batchSize, inputSize)
   local batchOutput = reservoir:forward(batchInput)
   mytester:asserteq(batchOutput:dim(), 2, 'batch output should be 2D')
   mytester:asserteq(batchOutput:size(1), batchSize, 'batch size mismatch')
   mytester:asserteq(batchOutput:size(2), outputSize, 'output size mismatch')
end

function ReservoirTest.test_options()
   -- Test reservoir with custom options
   local options = {
      spectralRadius = 0.95,
      inputScaling = 0.5,
      leakRate = 0.3,
      connectivity = 0.2,
      inputConnectivity = 0.3,
      seed = 42
   }
   
   local reservoir = nn.Reservoir(10, 50, 5, options)
   mytester:asserteq(reservoir.spectralRadius, 0.95, 'spectralRadius mismatch')
   mytester:asserteq(reservoir.inputScaling, 0.5, 'inputScaling mismatch')
   mytester:asserteq(reservoir.leakRate, 0.3, 'leakRate mismatch')
   mytester:asserteq(reservoir.connectivity, 0.2, 'connectivity mismatch')
   mytester:asserteq(reservoir.inputConnectivity, 0.3, 'inputConnectivity mismatch')
end

function ReservoirTest.test_gradient()
   -- Test gradient computation
   local inputSize = 3
   local reservoirSize = 10
   local outputSize = 2
   
   local reservoir = nn.Reservoir(inputSize, reservoirSize, outputSize)
   local input = torch.randn(inputSize)
   local gradOutput = torch.randn(outputSize)
   
   -- Forward
   local output = reservoir:forward(input)
   
   -- Backward
   reservoir:zeroGradParameters()
   local gradInput = reservoir:backward(input, gradOutput)
   
   -- Check gradient dimensions
   mytester:asserteq(gradInput:dim(), input:dim(), 'gradInput dimension mismatch')
   mytester:asserteq(gradInput:size(1), inputSize, 'gradInput size mismatch')
   
   -- Check parameters
   local params, gradParams = reservoir:parameters()
   mytester:asserteq(#params, 2, 'should have 2 parameter tensors')
   mytester:asserteq(#gradParams, 2, 'should have 2 gradient tensors')
end

function ReservoirTest.test_sequence()
   -- Test processing a sequence
   local inputSize = 4
   local reservoirSize = 15
   local outputSize = 1
   local seqlen = 10
   
   local reservoir = nn.Reservoir(inputSize, reservoirSize, outputSize)
   reservoir:forget()
   
   local outputs = {}
   for t = 1, seqlen do
      local input = torch.randn(inputSize)
      outputs[t] = reservoir:forward(input)
      mytester:asserteq(outputs[t]:size(1), outputSize, 'output size mismatch at step ' .. t)
   end
   
   -- State should be non-zero after processing sequence
   mytester:assert(reservoir.state:norm() > 0, 'state should be non-zero after sequence')
   
   -- Reset and check state is zero
   reservoir:forget()
   mytester:asserteq(reservoir.state:norm(), 0, 'state should be zero after forget')
end

function ReservoirTest.test_parameters_update()
   -- Test that only readout weights are trainable
   local reservoir = nn.Reservoir(5, 20, 3)
   
   -- Store initial weights
   local Win_initial = reservoir.Win:clone()
   local W_initial = reservoir.W:clone()
   local Wout_initial = reservoir.Wout:clone()
   
   -- Forward and backward
   local input = torch.randn(5)
   local gradOutput = torch.randn(3)
   reservoir:forward(input)
   reservoir:backward(input, gradOutput)
   reservoir:updateParameters(0.1)
   
   -- Win and W should not change (they are fixed)
   mytester:assertTensorEq(reservoir.Win, Win_initial, precision, 'Win should not change')
   mytester:assertTensorEq(reservoir.W, W_initial, precision, 'W should not change')
   
   -- Wout should change
   mytester:assert((reservoir.Wout - Wout_initial):abs():sum() > 0, 'Wout should change')
end

function ReservoirTest.test_deterministic()
   -- Test that same seed produces same results
   local options = { seed = 123 }
   local reservoir1 = nn.Reservoir(5, 20, 3, options)
   
   options = { seed = 123 }
   local reservoir2 = nn.Reservoir(5, 20, 3, options)
   
   -- Weights should be identical
   mytester:assertTensorEq(reservoir1.Win, reservoir2.Win, precision, 'Win should be identical with same seed')
   mytester:assertTensorEq(reservoir1.W, reservoir2.W, precision, 'W should be identical with same seed')
end

-- Run tests
mytester = torch.Tester()
mytester:add(ReservoirTest)
mytester:run()
