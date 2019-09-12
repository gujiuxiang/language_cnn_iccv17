-- Time-delayed Neural Network (i.e. 1-d CNN) with multiple filter widths
require 'cudnn'
require 'model.rnn.GaussianTransfer'
require 'model.rnn.BilinearD3_version2'
require 'model.rnn.CustomAlphaView'
local LSTM = require 'model.rnn.LSTM'
local GRU = require 'model.rnn.GRU'
local RNN = require 'model.rnn.RNN'
local RHN = require 'model.rnn.RHN'
require 'model.lm.memory'
local nninit = require 'nninit'

local LanguageCNN = {}

function LanguageCNN.Multimodal_GCNN(cap,state1, state2, state3, state4, state5)
  -- layer path
  local layer_conv11 = cudnn.TemporalConvolution(512, 512, 5)(cap) -- out: 12 x 512
  local layer_conv12 = cudnn.TemporalConvolution(512, 512, 5)(cap) -- out: 12 x 512
  local layer_conv13 = cudnn.TemporalConvolution(512, 512, 5)(cap) -- out: 12 x 512
  local layer_out1 = CNNRNN.ConvHighway(layer_conv11,layer_conv12,layer_conv13,state1,1)

  local layer_conv21 = cudnn.TemporalConvolution(512, 512, 5)(layer_out1) -- out: 8 x 512
  local layer_conv22 = cudnn.TemporalConvolution(512, 512, 5)(layer_out1) -- out: 8 x 512
  local layer_conv23 = cudnn.TemporalConvolution(512, 512, 5)(layer_out1) -- out: 8 x 512
  local layer_out2 = CNNRNN.ConvHighway(layer_conv21,layer_conv22,layer_conv23,state2,2)

  local layer_conv31 = cudnn.TemporalConvolution(512, 512, 3)(layer_out2) -- out: 6 x 512
  local layer_conv32 = cudnn.TemporalConvolution(512, 512, 3)(layer_out2) -- out: 6 x 512
  local layer_conv33 = cudnn.TemporalConvolution(512, 512, 3)(layer_out2) -- out: 6 x 512
  local layer_out3 = CNNRNN.ConvHighway(layer_conv31,layer_conv32,layer_conv33,state3,3)

  local layer_conv41 = cudnn.TemporalConvolution(512, 512, 3)(layer_out3) -- out: 4 x 512
  local layer_conv42 = cudnn.TemporalConvolution(512, 512, 3)(layer_out3) -- out: 4 x 512
  local layer_conv43 = cudnn.TemporalConvolution(512, 512, 3)(layer_out3) -- out: 4 x 512
  local layer_out4 = CNNRNN.ConvHighway(layer_conv41,layer_conv42,layer_conv43,state4,4)

  local layer_conv51 = cudnn.TemporalConvolution(512, 512, 3)(layer_out4) -- out: 2 x 512
  local layer_conv52 = cudnn.TemporalConvolution(512, 512, 3)(layer_out4) -- out: 2 x 512
  local layer_conv53 = cudnn.TemporalConvolution(512, 512, 3)(layer_out4) -- out: 2 x 512
  local layer_out5 = CNNRNN.ConvHighway(layer_conv51,layer_conv52,layer_conv53,state5,5)
  return layer_out1, layer_out2, layer_out3, layer_out4, layer_out5
end

function LanguageCNN.LanguageCNN_5Layers(cap, init)
  local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5)(cap)) -- out: 12 x 512
  local layer_conv2 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5)(layer_conv1)) -- out: 8 x 512
  local layer_conv3 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3)(layer_conv2)) -- out: 6 x 512
  local layer_conv4 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3)(layer_conv3)) -- out: 4 x 512
  local layer_conv5 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3)(layer_conv4)) -- out: 2 x 512
  return layer_conv5
end

function LanguageCNN.LanguageCNN_Highway_Gaussian_Gated(cap, init)
  local init_flag = init or -1
  if init_flag == -1 then
    local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5)(cap)) -- out: 12 x 512
    local layer_gate1 = nn.GaussianTransfer()(layer_conv1)
    local layer_gate1_out = nn.CMulTable()({layer_conv1,layer_gate1})
    local layer_gout1 = nn.CMulTable()({layer_gate1_out,layer_conv1})
    local layer_conv2 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5)(layer_gout1)) -- out: 8 x 512
    local layer_gate2 = nn.GaussianTransfer()(layer_conv2)
    local layer_gate2_out = nn.CMulTable()({layer_conv2,layer_gate2})
    local layer_gout2 = nn.CMulTable()({layer_gate2_out,layer_conv2})
    local layer_conv3 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3)(layer_gout2)) -- out: 6 x 512
    local layer_gate3 = nn.GaussianTransfer()(layer_conv3)
    local layer_gate3_out = nn.CMulTable()({layer_conv3,layer_gate3})
    local layer_gout3 = nn.CMulTable()({layer_gate3_out,layer_conv3})
    local layer_conv4 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3)(layer_gout3)) -- out: 4 x 512
    local layer_gate4 = nn.GaussianTransfer()(layer_conv4)
    local layer_gate4_out = nn.CMulTable()({layer_conv4,layer_gate4})
    local layer_gout4 = nn.CMulTable()({layer_gate4_out,layer_conv4})
    local layer_conv5 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3)(layer_gout4)) -- out: 2 x 512
    local layer_gate5 = nn.GaussianTransfer()(layer_conv5)
    local layer_gate5_out = nn.CMulTable()({layer_conv5,layer_gate5})
    local layer_gout5 = nn.CMulTable()({layer_gate5_out,layer_conv5})
    return layer_gout5
  else
    local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(cap)) -- out: 12 x 512
    local layer_gate1 = nn.GaussianTransfer()(layer_conv1)
    local layer_gate1_out = nn.CMulTable()({layer_conv1,layer_gate1})
    local layer_gout1 = nn.CMulTable()({layer_gate1_out,layer_conv1})
    local layer_conv2 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout1)) -- out: 8 x 512
    local layer_gate2 = nn.GaussianTransfer()(layer_conv2)
    local layer_gate2_out = nn.CMulTable()({layer_conv2,layer_gate2})
    local layer_gout2 = nn.CMulTable()({layer_gate2_out,layer_conv2})
    local layer_conv3 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout2)) -- out: 6 x 512
    local layer_gate3 = nn.GaussianTransfer()(layer_conv3)
    local layer_gate3_out = nn.CMulTable()({layer_conv3,layer_gate3})
    local layer_gout3 = nn.CMulTable()({layer_gate3_out,layer_conv3})
    local layer_conv4 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout3)) -- out: 4 x 512
    local layer_gate4 = nn.GaussianTransfer()(layer_conv4)
    local layer_gate4_out = nn.CMulTable()({layer_conv4,layer_gate4})
    local layer_gout4 = nn.CMulTable()({layer_gate4_out,layer_conv4})
    local layer_conv5 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout4)) -- out: 2 x 512
    local layer_gate5 = nn.GaussianTransfer()(layer_conv5)
    local layer_gate5_out = nn.CMulTable()({layer_conv5,layer_gate5})
    local layer_gout5 = nn.CMulTable()({layer_gate5_out,layer_conv5})
    return layer_gout5
  end
end
-------------------------------------------------------------------------------------
function LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_4In(cap, init)
  local init_flag = init or -1
  local layer_conv4 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(cap)) -- out: 2 x 512
  local layer_gate4 = nn.GaussianTransfer()(layer_conv4)
  local layer_gate4_out = nn.CMulTable()({layer_conv4,layer_gate4})
  local layer_gout4 = nn.CMulTable()({layer_gate4_out,layer_conv4})
  return layer_gout4
end

function LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_8In(cap, init)
  local init_flag = init or -1
  local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(cap)) -- out: 6 x 512
  local layer_gate1 = nn.GaussianTransfer()(layer_conv1)
  local layer_gate1_out = nn.CMulTable()({layer_conv1,layer_gate1})
  local layer_gout1 = nn.CMulTable()({layer_gate1_out,layer_conv1})
  local layer_conv2 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout1)) -- out: 4 x 512
  local layer_gate2 = nn.GaussianTransfer()(layer_conv2)
  local layer_gate2_out = nn.CMulTable()({layer_conv2,layer_gate2})
  local layer_gout2 = nn.CMulTable()({layer_gate2_out,layer_conv2})
  local layer_conv3 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout2)) -- out: 2 x 512
  local layer_gate3 = nn.GaussianTransfer()(layer_conv3)
  local layer_gate3_out = nn.CMulTable()({layer_conv3,layer_gate3})
  local layer_gout3 = nn.CMulTable()({layer_gate3_out,layer_conv3})
  return layer_gout3
end

function LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_4Layer(cap, init)
  local init_flag = init or -1
  local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(cap)) -- out: 12 x 512
  local layer_gate1 = nn.GaussianTransfer()(layer_conv1)
  local layer_gate1_out = nn.CMulTable()({layer_conv1,layer_gate1})
  local layer_gout1 = nn.CMulTable()({layer_gate1_out,layer_conv1})
  local layer_conv2 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout1)) -- out: 8 x 512
  local layer_gate2 = nn.GaussianTransfer()(layer_conv2)
  local layer_gate2_out = nn.CMulTable()({layer_conv2,layer_gate2})
  local layer_gout2 = nn.CMulTable()({layer_gate2_out,layer_conv2})
  local layer_conv3 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout2)) -- out: 4 x 512
  local layer_gate3 = nn.GaussianTransfer()(layer_conv3)
  local layer_gate3_out = nn.CMulTable()({layer_conv3,layer_gate3})
  local layer_gout3 = nn.CMulTable()({layer_gate3_out,layer_conv3})
  local layer_conv4 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout3)) -- out: 2 x 512
  local layer_gate4 = nn.GaussianTransfer()(layer_conv4)
  local layer_gate4_out = nn.CMulTable()({layer_conv4,layer_gate4})
  local layer_gout4 = nn.CMulTable()({layer_gate4_out,layer_conv4})
  return layer_gout4
end

function LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_4Layer_MaxPool(cap, init)
  local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(cap)) -- out: 12 x 512
  local layer_gate1 = nn.GaussianTransfer()(layer_conv1)
  local layer_gate1_out = nn.CMulTable()({layer_conv1,layer_gate1})
  local layer_gout1 = nn.CMulTable()({layer_gate1_out,layer_conv1})
  local layer_conv2 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout1)) -- out: 8 x 512
  local layer_gate2 = nn.GaussianTransfer()(layer_conv2)
  local layer_gate2_out = nn.CMulTable()({layer_conv2,layer_gate2})
  local layer_gout2 = nn.CMulTable()({layer_gate2_out,layer_conv2})
  local layer_conv3 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout2)) -- out: 6 x 512
  local layer_gate3 = nn.GaussianTransfer()(layer_conv3)
  local layer_gate3_out = nn.CMulTable()({layer_conv3,layer_gate3})
  local layer_gout4 = nn.TemporalMaxPooling(3, 1)(layer_gate3_out)
  local layer_conv5 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout4)) -- out: 2 x 512
  local layer_gate5 = nn.GaussianTransfer()(layer_conv5)
  local layer_gate5_out = nn.CMulTable()({layer_conv5,layer_gate5})
  local layer_gout5 = nn.CMulTable()({layer_gate5_out,layer_conv5})
  return layer_gout5
end

-------------------------------------------------------------------------------------

function LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_3Layer(cap, init)
  local init_flag = init or -1
  local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 7):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(cap)) -- out: 10 x 512
  local layer_gate1 = nn.GaussianTransfer()(layer_conv1)
  local layer_gate1_out = nn.CMulTable()({layer_conv1,layer_gate1})
  local layer_gout1 = nn.CMulTable()({layer_gate1_out,layer_conv1})
  local layer_conv2 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout1)) -- out: 6 x 512
  local layer_gate2 = nn.GaussianTransfer()(layer_conv2)
  local layer_gate2_out = nn.CMulTable()({layer_conv2,layer_gate2})
  local layer_gout2 = nn.CMulTable()({layer_gate2_out,layer_conv2})
  local layer_conv3 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout2)) -- out: 2 x 512
  local layer_gate3 = nn.GaussianTransfer()(layer_conv3)
  local layer_gate3_out = nn.CMulTable()({layer_conv3,layer_gate3})
  local layer_gout3 = nn.CMulTable()({layer_gate3_out,layer_conv3})
  return layer_gout3
end

function LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_3Layer_MaxPool(cap, init)
  local init_flag = init or -1
  local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(cap)) -- out: 12 x 512
  local layer_gate1 = nn.GaussianTransfer()(layer_conv1)
  local layer_gate1_out = nn.CMulTable()({layer_conv1,layer_gate1})
  local layer_gout1 = nn.CMulTable()({layer_gate1_out,layer_conv1})
  local layer_pool1 = nn.TemporalMaxPooling(3, 1)(layer_gout1) -- out: 10x512
  local layer_conv2 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_pool1)) -- out: 6 x 512
  local layer_gate2 = nn.GaussianTransfer()(layer_conv2)
  local layer_gate2_out = nn.CMulTable()({layer_conv2,layer_gate2})
  local layer_gout2 = nn.CMulTable()({layer_gate2_out,layer_conv2})
  local layer_pool2 = nn.TemporalMaxPooling(3, 1)(layer_gout2) -- out: 4x512
  local layer_conv4 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_pool2)) -- out: 2 x 512
  local layer_gate4 = nn.GaussianTransfer()(layer_conv4)
  local layer_gate4_out = nn.CMulTable()({layer_conv4,layer_gate4})
  local layer_gout4 = nn.CMulTable()({layer_gate4_out,layer_conv4})
  return layer_gout4
end

function LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_Filter33355(cap, init)
  local init_flag = init or -1
  local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(cap)) -- out: 14 x 512
  local layer_gate1 = nn.GaussianTransfer()(layer_conv1)
  local layer_gate1_out = nn.CMulTable()({layer_conv1,layer_gate1})
  local layer_gout1 = nn.CMulTable()({layer_gate1_out,layer_conv1})
  local layer_conv2 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout1)) -- out: 12 x 512
  local layer_gate2 = nn.GaussianTransfer()(layer_conv2)
  local layer_gate2_out = nn.CMulTable()({layer_conv2,layer_gate2})
  local layer_gout2 = nn.CMulTable()({layer_gate2_out,layer_conv2})
  local layer_conv3 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 3):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout2)) -- out: 10 x 512
  local layer_gate3 = nn.GaussianTransfer()(layer_conv3)
  local layer_gate3_out = nn.CMulTable()({layer_conv3,layer_gate3})
  local layer_gout3 = nn.CMulTable()({layer_gate3_out,layer_conv3})
  local layer_conv4 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout3)) -- out: 6 x 512
  local layer_gate4 = nn.GaussianTransfer()(layer_conv4)
  local layer_gate4_out = nn.CMulTable()({layer_conv4,layer_gate4})
  local layer_gout4 = nn.CMulTable()({layer_gate4_out,layer_conv4})
  local layer_conv5 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(layer_gout4)) -- out: 2 x 512
  local layer_gate5 = nn.GaussianTransfer()(layer_conv5)
  local layer_gate5_out = nn.CMulTable()({layer_conv5,layer_gate5})
  local layer_gout5 = nn.CMulTable()({layer_gate5_out,layer_conv5})
  return layer_gout5
end


function LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_Mean(cap, init)
  local init_flag = init or -1
  local layer_conv1 = nn.Threshold()(cudnn.TemporalConvolution(512, 512, 5):init('weight', nninit.addNormal, 0, 0.01):init('bias', nninit.constant, 0)(cap)) -- out: 12 x 512

  return layer_gout4
end


-------------------------------------------------------------------------------------

function LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local bias = bias or -2
  local cnn_Residual_out = cudnn.ReLU(true)(nn.Linear(1024, 512)(nn.Reshape(1024)(layer_conv5)))
  local Residual_out_trans= cudnn.ReLU(true)(nn.Linear(512, 512)(cnn_Residual_out))
  local cnn_transform_gate= cudnn.ReLU(true)(nn.AddConstant(bias)(nn.Linear(512, 512)(cnn_Residual_out)))
  local cnn_carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(cnn_transform_gate))
  local cap_output = nn.CAddTable()({nn.CMulTable()({cnn_transform_gate, Residual_out_trans}),nn.CMulTable()({cnn_carry_gate, cnn_Residual_out})})
  return cap_output
end

function LanguageCNN.LanguageCNN_16In()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 16, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end

function LanguageCNN.LanguageCNN_8In()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 8, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_8In(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end

function LanguageCNN.LanguageCNN_4In()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 4, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_4In(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end

function LanguageCNN.LanguageCNN_16In_33355()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 16, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_Filter33355(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end


function LanguageCNN.LanguageCNN_16In_4Layer()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 16, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_4Layer(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end

function LanguageCNN.LanguageCNN_16In_4Layer_MaxPool()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 16, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_4Layer_MaxPool(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end

function LanguageCNN.LanguageCNN_16In_3Layer()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 16, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_3Layer(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end

function LanguageCNN.LanguageCNN_16In_3Layer_MaxPool()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 16, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_3Layer_MaxPool(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end

function LanguageCNN.LanguageCNN_16In_Mean()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 16, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated_Mean(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end

function LanguageCNN.LanguageCNN_16In_NoIm()
  -- mscoco: vocab_size=9568, flickr30k=7051
  local bias = bias or -2

  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- curr_cap
  table.insert(inputs, nn.Identity()()) -- image

  local prev_cap = inputs[1]
  local curr_cap = inputs[2]
  local im_input = inputs[3]

  local cat_outputs = {}

  local cap = nn.View(torch.LongStorage{-1, 16, 512})(prev_cap)-- output bsx16x512
  -------------------------------------------------------------------------------------
  -- layer path
  --local layer_conv5 = LanguageCNN.LanguageCNN_5Layers(cap,1)
  local layer_conv5 = LanguageCNN.LanguageCNN_Highway_Gaussian_Gated(cap, 1)
  local cap_output = LanguageCNN.LanguageCNN_Highway(layer_conv5)
  local image_temp = cudnn.ReLU(true)(nn.Linear(512, 512)(nn.CAddTable()({im_input,cap_output})))
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(image_temp)))

  local multimodal_outputs = {}
  table.insert(multimodal_outputs, combine_out) -- first is lcnn output
  table.insert(multimodal_outputs, curr_cap) --second is current word
  local multimodal_outputs_concat = nn.JoinTable(2)(multimodal_outputs)
  table.insert(outputs, multimodal_outputs_concat)
  return nn.gModule(inputs, outputs)
end


function LanguageCNN.LanguageCNN_Gated()
  local inputs = {}
  local outputs = {}
  local cat_outputs = {}
  local bias = bias or -2
  table.insert(inputs, nn.Identity()()) -- captions
  table.insert(inputs, nn.Identity()()) -- s1
  table.insert(inputs, nn.Identity()()) -- s2
  table.insert(inputs, nn.Identity()()) -- s3
  table.insert(inputs, nn.Identity()()) -- s4
  table.insert(inputs, nn.Identity()()) -- s5
  local prev_cap = inputs[1]
  local state1 = inputs[2]
  local state2 = inputs[3]
  local state3 = inputs[4]
  local state4 = inputs[5]
  local state5 = inputs[6]
  local cap = nn.View(torch.LongStorage{-1, 16, 512})(prev_cap)-- output bsx16x512
  local stateo1, stateo2, stateo3, stateo4, stateo5 = LanguageCNN.Multimodal_GCNN(cap, state1, state2, state3, state4, state5)
  local cnn_Residual_out = nn.Sigmoid()(nn.Linear(1024, 512)(nn.Reshape(1024)(stateo5)))
  local drop_out = nn.Dropout(0.5)(cnn_Residual_out):annotate{name='drop_final'}
  local proj = nn.Linear(512, 9568)(drop_out):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)
  table.insert(outputs, stateo1)
  table.insert(outputs, stateo2)
  table.insert(outputs, stateo3)
  table.insert(outputs, stateo4)
  table.insert(outputs, stateo5)
  return nn.gModule(inputs, outputs)
end

return LanguageCNN
