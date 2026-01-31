package = "reservoirlua"
version = "scm-1"

source = {
   url = "git://github.com/torch/rnn",
   tag = "master"
}

description = {
   summary = "Reservoir Computing library for Torch (based on reservoirpy)",
   detailed = [[
A Lua implementation of Reservoir Computing (Echo State Networks) for Torch7,
inspired by the reservoirpy Python library. This module extends the rnn library
with efficient reservoir computing capabilities for time series prediction,
signal processing, and sequential data learning.

Features:
- Echo State Networks (ESN) with configurable parameters
- Spectral radius control for stable dynamics
- Sparse connectivity for efficiency
- Leak rate for temporal processing
- Only trains output layer for fast learning
- Compatible with existing rnn library modules
   ]],
   homepage = "https://github.com/torch/rnn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "rnn >= 1.0"
}

build = {
   type = "builtin",
   modules = {
      ['rnn.Reservoir'] = 'Reservoir.lua'
   }
}
