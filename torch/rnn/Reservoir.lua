------------------------------------------------------------------------
--[[ Reservoir ]]--
-- Implements a Reservoir Computing module (Echo State Network)
-- based on the reservoirpy Python library.
-- 
-- The reservoir is a randomly initialized recurrent layer with
-- fixed weights that are not trained. Only the output (readout) layer
-- is trained, making it very fast to train.
--
-- References:
-- - Lukoševičius, M. (2012). A Practical Guide to Applying Echo State Networks
-- - reservoirpy: https://github.com/reservoirpy/reservoirpy
------------------------------------------------------------------------
local Reservoir, parent = torch.class('nn.Reservoir', 'nn.Module')

function Reservoir:__init(inputSize, reservoirSize, outputSize, options)
   parent.__init(self)
   
   -- Default options
   options = options or {}
   local spectralRadius = options.spectralRadius or 0.9
   local inputScaling = options.inputScaling or 1.0
   local leakRate = options.leakRate or 1.0
   local connectivity = options.connectivity or 0.1
   local inputConnectivity = options.inputConnectivity or 0.1
   local seed = options.seed
   
   -- Store parameters
   self.inputSize = inputSize
   self.reservoirSize = reservoirSize
   self.outputSize = outputSize
   self.spectralRadius = spectralRadius
   self.inputScaling = inputScaling
   self.leakRate = leakRate
   self.connectivity = connectivity
   self.inputConnectivity = inputConnectivity
   
   -- Set random seed if provided
   if seed then
      torch.manualSeed(seed)
   end
   
   -- Initialize weights
   self:initializeWeights()
   
   -- Initialize state
   self.state = torch.Tensor()
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function Reservoir:initializeWeights()
   -- Initialize input weights (Win)
   self.Win = torch.Tensor(self.reservoirSize, self.inputSize)
   self.Win:uniform(-self.inputScaling, self.inputScaling)
   
   -- Apply input connectivity (sparsity)
   if self.inputConnectivity < 1.0 then
      local mask = torch.rand(self.reservoirSize, self.inputSize):lt(self.inputConnectivity)
      self.Win:cmul(mask:double())
   end
   
   -- Initialize reservoir weights (W) with sparsity
   self.W = torch.Tensor(self.reservoirSize, self.reservoirSize)
   self.W:uniform(-0.5, 0.5)
   
   -- Apply connectivity (sparsity)
   if self.connectivity < 1.0 then
      local mask = torch.rand(self.reservoirSize, self.reservoirSize):lt(self.connectivity)
      self.W:cmul(mask:double())
   end
   
   -- Scale to desired spectral radius
   self:scaleToSpectralRadius()
   
   -- Initialize output weights (Wout) - these will be trained
   self.Wout = torch.Tensor(self.outputSize, self.reservoirSize + self.inputSize)
   self.bias = torch.Tensor(self.outputSize)
   self.Wout:uniform(-0.1, 0.1)
   self.bias:zero()
   
   -- Gradients
   self.gradWout = torch.Tensor():resizeAs(self.Wout):zero()
   self.gradBias = torch.Tensor():resizeAs(self.bias):zero()
end

function Reservoir:scaleToSpectralRadius()
   -- Compute eigenvalues to get spectral radius
   -- Use power iteration method for efficiency
   local v = torch.randn(self.reservoirSize, 1)
   v:div(v:norm())
   
   for i = 1, 100 do
      v = self.W * v
      local norm = v:norm()
      v:div(norm)
   end
   
   -- Compute the approximate largest eigenvalue
   local lambda = (self.W * v):norm()
   
   -- Scale W to desired spectral radius
   if lambda > 0 then
      self.W:mul(self.spectralRadius / lambda)
   end
end

function Reservoir:updateOutput(input)
   local batchSize
   if input:dim() == 1 then
      -- Single sample
      batchSize = nil
   elseif input:dim() == 2 then
      -- Batch mode
      batchSize = input:size(1)
   else
      error('Input must be 1D or 2D tensor')
   end
   
   -- Initialize state if needed
   if self.state:nElement() == 0 then
      if batchSize then
         self.state:resize(batchSize, self.reservoirSize):zero()
      else
         self.state:resize(self.reservoirSize):zero()
      end
   end
   
   -- Ensure state has correct size
   if batchSize then
      if self.state:dim() == 1 then
         self.state = self.state:view(1, -1):expandAs(torch.Tensor(batchSize, self.reservoirSize))
      end
   else
      if self.state:dim() == 2 then
         self.state = self.state[1]
      end
   end
   
   -- Update reservoir state (fixed weights, not trained)
   if batchSize then
      -- Batch processing
      self.output:resize(batchSize, self.outputSize)
      local newState = torch.Tensor(batchSize, self.reservoirSize)
      
      for i = 1, batchSize do
         local sampleInput = input[i]
         local sampleState = self.state[i]
         local inputContribution = self.Win * sampleInput:view(self.inputSize, 1)
         local recurrentContribution = self.W * sampleState:view(self.reservoirSize, 1)
         local preActivation = inputContribution + recurrentContribution
         
         -- Apply tanh activation
         local activation = torch.tanh(preActivation)
         
         -- Apply leak rate
         newState[i]:copy(sampleState)
         newState[i]:mul(1 - self.leakRate)
         newState[i]:add(self.leakRate, activation:view(-1))
         
         -- Compute output using trainable weights
         local extendedState = torch.cat(sampleInput:view(-1), newState[i]:view(-1))
         self.output[i]:copy((self.Wout * extendedState:view(-1, 1) + self.bias:view(-1, 1)):view(-1))
      end
      
      self.state:copy(newState)
   else
      -- Single sample processing
      local inputContribution = self.Win * input:view(self.inputSize, 1)
      local recurrentContribution = self.W * self.state:view(self.reservoirSize, 1)
      local preActivation = inputContribution + recurrentContribution
      
      -- Apply tanh activation
      local activation = torch.tanh(preActivation)
      
      -- Apply leak rate
      local newState = self.state:clone()
      newState:mul(1 - self.leakRate)
      newState:add(self.leakRate, activation:view(-1))
      
      -- Compute output using trainable weights
      local extendedState = torch.cat(input:view(-1), newState:view(-1))
      self.output = (self.Wout * extendedState:view(-1, 1) + self.bias:view(-1, 1)):view(-1)
      
      self.state:copy(newState)
   end
   
   return self.output
end

function Reservoir:updateGradInput(input, gradOutput)
   -- For reservoir computing, we typically only train the output layer
   -- The reservoir weights (Win, W) are fixed, so gradients w.r.t. input are not computed
   self.gradInput:resizeAs(input):zero()
   return self.gradInput
end

function Reservoir:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   local batchSize
   if input:dim() == 1 then
      batchSize = nil
   elseif input:dim() == 2 then
      batchSize = input:size(1)
   end
   
   if batchSize then
      -- Batch mode
      for i = 1, batchSize do
         local extendedState = torch.cat(input[i]:view(-1), self.state[i]:view(-1))
         self.gradWout:addr(scale, gradOutput[i]:view(-1, 1), extendedState:view(1, -1))
         self.gradBias:add(scale, gradOutput[i]:view(-1))
      end
   else
      -- Single sample
      local extendedState = torch.cat(input:view(-1), self.state:view(-1))
      self.gradWout:addr(scale, gradOutput:view(-1, 1), extendedState:view(1, -1))
      self.gradBias:add(scale, gradOutput:view(-1))
   end
end

function Reservoir:reset(stdv)
   -- Reset the reservoir state
   self.state:zero()
end

function Reservoir:forget()
   -- Alias for reset, used in rnn package
   self:reset()
end

function Reservoir:updateParameters(learningRate)
   self.Wout:add(-learningRate, self.gradWout)
   self.bias:add(-learningRate, self.gradBias)
end

function Reservoir:zeroGradParameters()
   self.gradWout:zero()
   self.gradBias:zero()
end

function Reservoir:parameters()
   return {self.Wout, self.bias}, {self.gradWout, self.gradBias}
end

function Reservoir:__tostring__()
   return string.format('%s(%d -> %d -> %d, SR=%.2f, LR=%.2f)', 
      torch.type(self), self.inputSize, self.reservoirSize, 
      self.outputSize, self.spectralRadius, self.leakRate)
end

-- Backward compatibility
nn.ESN = nn.Reservoir
