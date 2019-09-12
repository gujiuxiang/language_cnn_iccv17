require 'nn'
require 'nngraph'

require 'model.rnn.LinearTensorD3'
require 'model.rnn.BilinearD3_version2'
require 'model.rnn.CAddTableD2D3'
require 'model.rnn.CustomAlphaView'

require 'model.rnn.probe' -- for debugger on nngraph module, put the layer to check gradient and outputs
require 'model.rnn.utils_bg' -- also for debugger purpose

local NORNN = {}

function NORNN.rnn(opt)
  opt.dropout = opt.dropout or 0
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  local i2h,next_h
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
    -- RNN tick
    i2h = nn.Linear(input_size_L, opt.rnn_size*2)(x)
    next_h = prev_h
    
    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = i2h
  if opt.dropout > 0 then top_h = nn.Dropout(opt.dropout)(top_h) end
  local proj = nn.Linear(2*opt.rnn_size, opt.output_size)(top_h):annotate{name='drop_final'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function NORNN.rnn_add(opt)
  opt.dropout = opt.dropout or 0
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  local i2h, next_w
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
    -- RNN tick
    i2h = nn.Linear(input_size_L, opt.rnn_size)(x)
    local multi_outputs = {}
    table.insert(multi_outputs, prev_h) -- first is lcnn output
    table.insert(multi_outputs, i2h) --second is current word
    prev_w = nn.Linear(input_size_L, opt.rnn_size)(nn.JoinTable(2)(multi_outputs))
    
  end
-- set up the decoder
  local top_h = prev_w
  if opt.dropout > 0 then top_h = nn.Dropout(opt.dropout)(top_h) end
  local proj = nn.Linear(opt.rnn_size, opt.output_size)(top_h):annotate{name='drop_final'}
  local next_h = nn.Linear(opt.output_size, opt.rnn_size)(proj)
  table.insert(outputs, next_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return NORNN
