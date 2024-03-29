--[[
   This file implements Gate Pooling as described in the paper:
   "A Convolutional Architecture for Word Sequence Prediction"
                   by Mingxuan Wang; Zhengdong Lu; Hang Li; Wenbin Jiang; Qun Liu
   
   The operation is using the gating to select the values out of a sequence.
   k can be a calculated value or pre-defined
   The value of k can be calulated as in the paper by using:
      k_top as minK
      (L-l)/L as factor
   
   Where:
      k_top is the desired sequence length at the end of the convolution part,
      L is the total number of layers,
      l is this layers number
]]

local CNNGatePooling, parent = torch.class('nn.CNNGatePooling', 'nn.Module')

function CNNGatePooling:__init(minK, factor)
   parent.__init(self)
   
   self.factor = factor or 0
end

function CNNGatePooling:updateOutput(input)
   assert(input:dim() == 2 or input:dim() == 3, 'Only 2D or 3D(batch mode) accepted')

   local seqDim = input:dim()-1
   local k = math.max(self.minK, math.ceil(self.factor*input:size(seqDim)))
   assert(input:size(seqDim) >= self.minK, 'Input sequence length (' .. input:size(seqDim) .. ') too small for desired k value (' .. k .. ')')

   -- Sort input in descending order
   local sorted, allIndices = input:sort(seqDim,true)
   -- Reduce the indices to only include the top-k and return to original order by sorting
   self.indices = allIndices:narrow(seqDim, 1, k):sort(seqDim)

   self.output = input:gather(seqDim, self.indices)

   return self.output
end

function CNNGatePooling:updateGradInput(input, gradOutput)
   if self.gradInput then
      local seqDim = input:dim()-1

      self.gradInput:resizeAs(input)
      self.gradInput:zero()

      -- Using the previously stored indices, add the gradOutputs to their respective
      -- input indices in the self.gradInput buffer
      local updateValues = self.gradInput:gather(seqDim, self.indices)
      updateValues:add(gradOutput)
      self.gradInput:scatter(seqDim, self.indices, updateValues)

      return self.gradInput
   end
end

function CNNGatePooling:clearState()
   nn.utils.clear(self, 'indices')
   return parent.clearState(self)
end