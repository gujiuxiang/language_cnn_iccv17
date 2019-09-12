require 'nn'
require 'nngraph'

require 'model.rnn.LinearTensorD3'
require 'model.rnn.BilinearD3_version2'
require 'model.rnn.CAddTableD2D3'
require 'model.rnn.CustomAlphaView'

require 'model.rnn.probe' -- for debugger on nngraph module, put the layer to check gradient and outputs
require 'model.rnn.utils_bg' -- also for debugger purpose

local GRU = {}
--[[
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
function GRU.gru(opt)
  opt.dropout = opt.dropout or 0
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  function new_input_sum(insize, rnn_size, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,opt.num_layers do

    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = opt.input_size
    else
      x = outputs[(L-1)]
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x) end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, opt.rnn_size, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, opt.rnn_size, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(opt.rnn_size, opt.rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, opt.rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
  -- set up the decoder
  local top_h = outputs[#outputs]
  if opt.dropout > 0 then top_h = nn.Dropout(opt.dropout)(top_h) end
  local proj = nn.Linear(opt.rnn_size, opt.output_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function GRU.makeWeightedSumUnit()
  -- note each sample in the batch may has different alignments(or called weights)
  local alpha = nn.Identity()() -- bz * L
  local alphaMatrix = nn.CustomAlphaView()(alpha) -- bz * L * 1

  local x = nn.Identity()() -- bz * L * xDim
  local g = nn.MM(true, false)({x, alphaMatrix}) -- bz * xDim * 1

  g = nn.Select(3, 1)(g) -- bz * xDim
  local inputs, outputs = {x, alpha}, {g}

  -- return a nn.Module
  return nn.gModule(inputs, outputs)
end

-- pass testing
-- input_size
function GRU.gru_withou_decoder(opt)
  opt.dropout = opt.dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) --x, for test, bz * 8
  for L = 1,opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L], bz * 8
  end

  local x, input_size_L

  local outputs = {}
  for L = 1,opt.num_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = opt.input_size
    else -- currently only 1 layer, this is not modified
      x = outputs[L-1] -- lower layer output: next_h
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, opt.rnn_size, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, opt.rnn_size, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(opt.rnn_size, opt.rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, opt.rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
  -- inputs = {x , prev_c, prev_h}
  -- outputs = {next_c, next_h}
  return nn.gModule(inputs, outputs)
end

-- not pass testing ??????
function GRU.with_output_attention(opt)
  opt.dropout = opt.dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) --x, glimpse, for test, bz * 8
  table.insert(inputs, nn.Identity()()) -- As

  table.insert(inputs, nn.Identity()()) -- prev_c[L], bz * 8

  local x, input_size_L
  function new_input_sum(insize, rnn_size, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end
  local outputs = {}
  for L = 1,opt.num_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = opt.input_size
    else -- currently only 1 layer, this is not modified
      x = outputs[(L-1)*2] -- lower layer output: next_h
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, opt.rnn_size, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, opt.rnn_size, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(opt.rnn_size, opt.rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, opt.rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local prev_h = nn.CAddTable()({zh, zhm1})

    -- gated cells form the output
    -- next is 'current', which will be used as input at the next timestep
    table.insert(outputs, prev_h)
  end
  -- inputs = {x , prev_c, prev_h}
  -- outputs = {next_c, next_h, logsoft}
  -- set up output
  local top_h = outputs[#outputs]

  local logsoft = GRU.Make_Output_Attention_Bilinear_Unit(opt.rnn_size, opt.word_embed_size, opt.output_size)({top_h, inputs[2]})
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function GRU.Make_Input_Attention_Bilinear_Unit(word_embed_size, word_embed_size, m)

  local prev_word_embed = nn.Identity()() -- prev_word_embed: bz * 300
  local As = nn.Identity()() -- bz * 16 * 300

  -- the number of attributes, we may change it to be 10
  local attention_output = nn.BilinearD3(word_embed_size, word_embed_size, 10, false)({prev_word_embed, As}) -- no bias
  -- attention_output = nn.Probe()(attention_output)
  local alpha = nn.SoftMax()(attention_output) -- bz * L

  local g_in = GRU.makeWeightedSumUnit()({As, alpha}) -- g_in: bz * 300

  -- local temp = nn.CAddTable()({nn.CMul(word_embed_size)(g_in), prev_word_embed})
  local temp = nn.CAddTable()({g_in, prev_word_embed})

  local x_t = nn.Linear(word_embed_size, m)(temp) -- m is 512 for coco, xt: bz * 512

  local inputs, outputs = {prev_word_embed, As}, {x_t}

  return nn.gModule(inputs, outputs)
end

function GRU.Make_Output_Attention_Bilinear_Unit(hDim, word_embed_size, outputSize, dropout)
  dropout = dropout or 0
  local h_t = nn.Identity()() -- current h_t: bz * 512
  local As = nn.Identity()() -- bz * 10 * 300
  -- the number of attributes, we may change it to be 10
  local attention_output = nn.BilinearD3(hDim, word_embed_size, 10, false)({h_t, nn.Tanh()(As)}) -- no bias
  -- attention_output = nn.Probe()(attention_output)
  local beta = nn.SoftMax()(attention_output) -- bz * L

  local g_out = GRU.makeWeightedSumUnit()({nn.Tanh()(As), beta}) -- g_out: bz * 300(d)

  local temp = nn.CAddTable()({nn.Linear(word_embed_size, hDim)(nn.CMul(word_embed_size)(g_out)), h_t})

  if dropout > 0 then temp = nn.Dropout(dropout)(temp) end

  local proj = nn.Linear(hDim, outputSize)(temp) -- proj: bz * outputSize
  local logsoft = nn.LogSoftMax()(proj)

  local inputs, outputs = {h_t, As}, {logsoft}

  return nn.gModule(inputs, outputs)
end

return GRU
