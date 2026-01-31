require 'rnn'

-- This example demonstrates the Reservoir Computing (Echo State Network) module
-- based on the reservoirpy Python library

print('=== Reservoir Computing (ESN) Example ===')
print('')

-- Hyper-parameters
local inputSize = 3
local reservoirSize = 100
local outputSize = 1
local seqlen = 50
local batchSize = 8
local lr = 0.01
local iterations = 100

-- Create a Reservoir (Echo State Network) with options
local options = {
   spectralRadius = 0.9,     -- Controls dynamics of reservoir
   inputScaling = 1.0,       -- Scaling of input weights
   leakRate = 0.3,           -- Leak rate (alpha) for leaky integrator
   connectivity = 0.1,       -- Sparsity of reservoir weights
   inputConnectivity = 0.2,  -- Sparsity of input weights
   seed = 12345              -- Random seed for reproducibility
}

local reservoir = nn.Reservoir(inputSize, reservoirSize, outputSize, options)

print('Created reservoir:')
print(reservoir)
print('')

-- Build criterion (Mean Squared Error for regression)
local criterion = nn.MSECriterion()

-- Generate synthetic time series data
-- Task: predict next value in a sine wave
local function generateData(length, batchSize)
   local inputs = {}
   local targets = {}
   
   for t = 1, length do
      local input = torch.Tensor(batchSize, inputSize)
      local target = torch.Tensor(batchSize, outputSize)
      
      for b = 1, batchSize do
         -- Generate sine wave with some noise
         local phase = (t + b * 10) * 0.1
         input[b][1] = math.sin(phase)
         input[b][2] = math.cos(phase)
         input[b][3] = math.sin(2 * phase)
         
         -- Target is next value
         local nextPhase = (t + 1 + b * 10) * 0.1
         target[b][1] = math.sin(nextPhase)
      end
      
      inputs[t] = input
      targets[t] = target
   end
   
   return inputs, targets
end

print('Training Reservoir...')
print('')

-- Training loop
for iter = 1, iterations do
   -- Generate fresh data for each iteration
   local inputs, targets = generateData(seqlen, batchSize)
   
   -- Reset reservoir state for new sequence
   reservoir:forget()
   reservoir:zeroGradParameters()
   
   local totalLoss = 0
   
   -- Forward pass
   for t = 1, seqlen do
      local output = reservoir:forward(inputs[t])
      local loss = criterion:forward(output, targets[t])
      totalLoss = totalLoss + loss
      
      -- Backward pass
      local gradOutput = criterion:backward(output, targets[t])
      reservoir:backward(inputs[t], gradOutput)
   end
   
   -- Update parameters (only readout layer is trained)
   reservoir:updateParameters(lr)
   
   -- Print progress
   if iter % 10 == 0 or iter == 1 then
      print(string.format('Iteration %d: Average Loss = %.6f', iter, totalLoss / seqlen))
   end
end

print('')
print('Training completed!')
print('')

-- Test the trained reservoir
print('Testing on new sequence...')
reservoir:forget()

local testInputs, testTargets = generateData(20, 1)
local testOutputs = {}

for t = 1, 20 do
   testOutputs[t] = reservoir:forward(testInputs[t])
end

print('')
print('Sample predictions (first 10 steps):')
print('Step | Target  | Prediction | Error')
print('-----|---------|------------|-------')
for t = 1, math.min(10, 20) do
   local target = testTargets[t][1][1]
   local pred = testOutputs[t][1][1]
   local error = math.abs(target - pred)
   print(string.format('%4d | %7.4f | %10.4f | %.4f', t, target, pred, error))
end

print('')
print('=== Example completed ===')
