require 'nn'
require 'nngraph'

require 'model.rnn.GaussianTransfer'
require 'model.rnn.AddTransfer'

require 'model.rnn.LinearTensorD3'
require 'model.rnn.BilinearD3_version2'
require 'model.rnn.CAddTableD2D3'
require 'model.rnn.CustomAlphaView'

require 'model.rnn.probe' -- for debugger on nngraph module, put the layer to check gradient and outputs
require 'model.rnn.utils_bg' -- also for debugger purpose

local LSTM = {}

function LSTM.encoderstlstm3(opt) -- n: num_layers
  opt.dropout = opt.dropout or 0

  -- there will be 4*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1, opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_cj[L]
    table.insert(inputs, nn.Identity()()) -- prev_hj[L]
  end
  for L = 1, opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_ct[L]
    table.insert(inputs, nn.Identity()()) -- prev_ht[L]
  end

  local x, input_size_L
  local outputs = {}

  for L = 1, opt.num_layers do
    -- c,h from previos steps
    local prev_cj = inputs[L*2]
    local prev_hj = inputs[L*2+1]

    local prev_ct = inputs[n*2+L*2]
    local prev_ht = inputs[n*2+L*2+1]

    -- the input to this layer
    if (L == 1) then
      x = inputs[1]
      input_size_L = opt.input_size
    else
      x = outputs[(L-1)*2]
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x) end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end

    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 5 * opt.rnn_size)(x):annotate{ name = 'i2h_' .. L}
    local h2hj = nn.Linear(opt.rnn_size, 5 * opt.rnn_size)(prev_hj):annotate{name = 'h2hj_' .. L}
    local h2ht = nn.Linear(opt.rnn_size, 5 * opt.rnn_size)(prev_ht):annotate{name = 'h2ht_' .. L}
    local all_input_sums = nn.CAddTable()({i2h, h2hj, h2ht})

    local reshaped = nn.Reshape(5, rnn_size)(all_input_sums)
    local n1, n2, n3, n4, n5 = nn.SplitTable(2)(reshaped):split(5)

    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate_j = nn.Sigmoid()(n2)
    local forget_gate_t = nn.Sigmoid()(n3)
    local out_gate = nn.Sigmoid()(n4)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n5)

    local h2pj = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_hj):annotate{name = 'h2pj_' .. L}
    local h2pt = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_ht):annotate{name = 'h2pt_' .. L}
    local all_predict_sums = nn.CAddTable()({h2pj, h2pt})
    local predict_transform = nn.Tanh()(all_predict_sums)

    local trust_gate = nn.GaussianTransfer()( nn.CSubTable()({in_transform, predict_transform}) ):annotate{name = 'trustgate_' .. L}

    local new_in_gate = nn.CMulTable()({ in_gate, trust_gate }):annotate{name = 'newingate_' .. L}
    local new_forget_gate_j = nn.CMulTable()({ forget_gate_j, nn.AddTransfer(-1, true)(trust_gate) }):annotate{name = 'newforgetgatej_' .. L}
    local new_forget_gate_t = nn.CMulTable()({ forget_gate_t, nn.AddTransfer(-1, true)(trust_gate) }):annotate{name = 'newforgetgatet_' .. L}

    -- perform the STLSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({new_forget_gate_j, prev_cj}),
        nn.CMulTable()({new_forget_gate_t, prev_ct}),
        nn.CMulTable()({new_in_gate, in_transform}) })

    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  return nn.gModule(inputs, outputs)
end

function LSTM.GLSTM(opt)
  opt.dropout = opt.dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,opt.num_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = opt.input_size
    else
      x = outputs[(L-1)*2]
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * opt.rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, opt.rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    local in_transform = nn.Tanh()(n4)
    -- decode the write inputs

    local h2p = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h):annotate{name = 'h2p_' .. L}
    local predict_transform = nn.Tanh()(h2p)
    local trust_gate = nn.GaussianTransfer()( nn.CSubTable()({in_transform, predict_transform}) ):annotate{name = 'trustgate_' .. L}

    local new_in_gate = nn.CMulTable()({ in_gate, trust_gate }):annotate{name = 'newingate_' .. L}
    local new_forget_gate = nn.CMulTable()({ forget_gate, nn.AddTransfer(-1, true)(trust_gate) }):annotate{name = 'newforgetgatej_' .. L}
    -- perform the LSTM update
    local next_c = nn.CAddTable()({nn.CMulTable()({new_forget_gate, prev_c}), nn.CMulTable()({new_in_gate,in_transform})})
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if opt.dropout > 0 then top_h = nn.Dropout(opt.dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(opt.rnn_size, opt.output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function LSTM.makeWeightedSumUnit()
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

function LSTM.lstm(opt)
  opt.dropout = opt.dropout or 0
  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,opt.num_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = opt.input_size
    else
      x = outputs[(L-1)*2]
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * opt.rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, opt.rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c}), nn.CMulTable()({in_gate,in_transform})})
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if opt.dropout > 0 then top_h = nn.Dropout(opt.dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(opt.rnn_size, opt.output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function LSTM.without_output_decoder(opt)
  opt.dropout = opt.dropout or 0
  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,opt.num_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = opt.input_size
    else
      x = outputs[(L-1)*2]
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * opt.rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, opt.rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c}), nn.CMulTable()({in_gate,in_transform})})
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  return nn.gModule(inputs, outputs)
end

function LSTM.with_output_attention(opt)
  opt.dropout = opt.dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) --x, glimpse, for test, bz * 8
  table.insert(inputs, nn.Identity()()) -- As

  for L = 1,opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L], bz * 8
    table.insert(inputs, nn.Identity()()) -- prev_h[L], bz * 8
  end

  local x, input_size_L

  local outputs = {}
  for L = 1,opt.num_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+2]
    local prev_c = inputs[L*2+1]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = opt.input_size
    else -- currently only 1 layer, this is not modified
      x = outputs[(L-1)*2] -- lower layer output: next_h
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end

    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * opt.rnn_size)(x):annotate{name='w2h_'..L}
    -- to avoid double bias terms
    local h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size, false)(prev_h):annotate{name='h2h_'..L}

    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, opt.rnn_size)(all_input_sums)

    -- 2 instead of 1 because it supports batch input
    -- split method is a node method which will return 4 new nodes
    -- because nn.SplitTable(2)() will return 4 output nodes
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    -- next is 'current', which will be used as input at the next timestep
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  -- inputs = {x , prev_c, prev_h}
  -- outputs = {next_c, next_h, logsoft}
  -- set up output
  local top_h = outputs[#outputs]

  local logsoft = LSTM.Make_Output_Attention_Bilinear_Unit(opt.rnn_size, opt.word_embed_size, opt.output_size)({top_h, inputs[2]})
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function LSTM.Make_Input_Attention_Bilinear_Unit(word_embed_size, word_embed_size, m)

  local prev_word_embed = nn.Identity()() -- prev_word_embed: bz * 300
  local As = nn.Identity()() -- bz * 16 * 300

  -- the number of attributes, we may change it to be 10
  local attention_output = nn.BilinearD3(word_embed_size, word_embed_size, 10, false)({prev_word_embed, As}) -- no bias
  -- attention_output = nn.Probe()(attention_output)
  local alpha = nn.SoftMax()(attention_output) -- bz * L

  local g_in = LSTM.makeWeightedSumUnit()({As, alpha}) -- g_in: bz * 300

  -- local temp = nn.CAddTable()({nn.CMul(word_embed_size)(g_in), prev_word_embed})
  local temp = nn.CAddTable()({g_in, prev_word_embed})

  local x_t = nn.Linear(word_embed_size, m)(temp) -- m is 512 for coco, xt: bz * 512

  local inputs, outputs = {prev_word_embed, As}, {x_t}

  return nn.gModule(inputs, outputs)
end

function LSTM.Make_Output_Attention_Bilinear_Unit(hDim, word_embed_size, outputSize, dropout)
  dropout = dropout or 0
  local h_t = nn.Identity()() -- current h_t: bz * 512
  local As = nn.Identity()() -- bz * 10 * 300
  -- the number of attributes, we may change it to be 10
  local attention_output = nn.BilinearD3(hDim, word_embed_size, 10, false)({h_t, nn.Tanh()(As)}) -- no bias
  -- attention_output = nn.Probe()(attention_output)
  local beta = nn.SoftMax()(attention_output) -- bz * L

  local g_out = LSTM.makeWeightedSumUnit()({nn.Tanh()(As), beta}) -- g_out: bz * 300(d)
  --CMul: Applies a component-wise multiplication to the incoming data, i.e. y_i = w_i * x_i . For example, nn.CMul(3,4,5) is equivalent to nn.CMul(torch.LongStorage{3,4,5}) 
  local temp = nn.CAddTable()({nn.Linear(word_embed_size, hDim)(nn.CMul(word_embed_size)(g_out)), h_t})

  if dropout > 0 then temp = nn.Dropout(dropout)(temp) end

  local proj = nn.Linear(hDim, outputSize)(temp) -- proj: bz * outputSize
  local logsoft = nn.LogSoftMax()(proj)

  local inputs, outputs = {h_t, As}, {logsoft}

  return nn.gModule(inputs, outputs)
end

function LSTM.M_LSTM(opt)
  opt.dropout = opt.dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  table.insert(inputs, nn.Identity()()) -- previous char-cnn input
  table.insert(inputs, nn.Identity()()) -- previous image

  local x, input_size_L
  local outputs = {}
  for L = 1,opt.num_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = opt.input_size
    else
      x = outputs[(L-1)*2]
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * opt.rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, opt.rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c}), nn.CMulTable()({in_gate,in_transform})})
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]

  local prev_word = inputs[2*n+2]
  local image = inputs[2*n+2+1]
  local prev_word_map = nn.Linear(512, 512)(prev_word)
  local image_map = nn.Linear(512, 512)(image)
  local combine = nn.CAddTable()({top_h, nn.CAddTable()({prev_word_map, image_map})})
  local combine_out = nn.MulConstant(1.7159)(nn.Tanh()(nn.MulConstant(2/3)(combine)))

  if opt.dropout > 0 then combine_out = nn.Dropout(opt.dropout)(combine_out):annotate{name='drop_final'} end
  local proj = nn.Linear(opt.rnn_size, opt.output_size)(combine_out):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function LSTM.BNLSTM(opt)
  opt.dropout = opt.dropout or 0
  local bn = 0
  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1, opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1, opt.num_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[L * 2 + 1]
    local prev_c = inputs[L * 2]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = opt.input_size
    else
      x = outputs[(L - 1) * 2]
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x) end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end
    -- recurrent batch normalization
    -- http://arxiv.org/abs/1603.09025
    local bn_wx, bn_wh, bn_c
    if bn then
      bn_wx = nn.BatchNormalization(4 * opt.rnn_size, 1e-5, 0.1, true)
      bn_wh = nn.BatchNormalization(4 * opt.rnn_size, 1e-5, 0.1, true)
      bn_c = nn.BatchNormalization(opt.rnn_size, 1e-5, 0.1, true)

      -- initialise beta=0, gamma=0.1
      bn_wx.weight:fill(0.1)
      bn_wx.bias:zero()
      bn_wh.weight:fill(0.1)
      bn_wh.bias:zero()
      bn_c.weight:fill(0.1)
      bn_c.bias:zero()
    else
      bn_wx = nn.Identity()
      bn_wh = nn.Identity()
      bn_c = nn.Identity()
    end
    -- evaluate the input sums at once for efficiency
    local i2h = bn_wx(nn.Linear(input_size_L, 4 * opt.rnn_size)(x):annotate { name = 'i2h_' .. L }):annotate { name = 'bn_wx_' .. L }
    local h2h = bn_wh(nn.Linear(opt.rnn_size, 4 * opt.rnn_size, false)(prev_h):annotate { name = 'h2h_' .. L }):annotate { name = 'bn_wh_' .. L }
    local all_input_sums = nn.CAddTable()({ i2h, h2h })

    local reshaped = nn.Reshape(4, opt.rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({ forget_gate, prev_c }),
        nn.CMulTable()({ in_gate, in_transform })
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({ out_gate, nn.Tanh()(bn_c(next_c):annotate { name = 'bn_c_' .. L }) })

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- nngraph.annotateNodes()
  -- set up the decoder

  local top_h = outputs[#outputs]
  local proj = nn.Linear(opt.rnn_size, opt.output_size)(top_h):annotate{name='decoder'}
  if opt.dropout > 0 then top_h = nn.Dropout(opt.dropout)(top_h):annotate{name='drop_final'} end
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function LSTM.ArrayLSTM(opt)
  opt.dropout = opt.dropout or 0
  local cell_size = 4
  -- there will be 5*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,opt.num_layers do
    for S = 1,cell_size do
      table.insert(inputs, nn.Identity()()) -- prev_c_[S][L]. S is from 1 to 4
    end
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,opt.num_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[(L-1)*(cell_size+1)+(cell_size+2)]
    local prev_c = {}
    for C = 1,cell_size do
      table.insert(prev_c, inputs[(L-1)*(cell_size+1)+C+1])
    end
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      -- x = OneHot(input_size)(inputs[1])
      input_size_L = opt.input_size
    else
      x = outputs[(L-1)*(cell_size+1)]
      if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = opt.rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local next_c = {}
    local next_h = {}
    for C =1,cell_size do
      local i2h = nn.Linear(input_size_L, cell_size * opt.rnn_size)(x):annotate{name='i2h_'..L}
      local h2h = nn.Linear(opt.rnn_size, cell_size * opt.rnn_size)(prev_h):annotate{name='h2h_'..L}
      local all_input_sums = nn.CAddTable()({i2h, h2h})
      local reshaped = nn.Reshape(cell_size, opt.rnn_size)(all_input_sums)
      local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
      -- decode the gates
      local in_gate = nn.Sigmoid()(n1)
      local forget_gate = nn.Sigmoid()(n2)
      local out_gate = nn.Sigmoid()(n3)
      -- decode the write inputs
      local in_transform = nn.Tanh()(n4)
      -- perform the LSTM update
      table.insert(next_c, nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c[C]}),nn.CMulTable()({in_gate, in_transform})}))
      -- gated cells form the output
      table.insert(next_h, nn.CMulTable()({out_gate, nn.Tanh()(next_c[C])}))
    end
    local next_h_sum = next_h[1]
    for C = 2,cell_size do
      next_h_sum = nn.CAddTable()({next_h_sum, next_h[C]})
    end

    for C = 1,cell_size do
      table.insert(outputs, next_c[C])
    end
    table.insert(outputs, next_h_sum)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if opt.dropout > 0 then top_h = nn.Dropout(opt.dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(opt.rnn_size, opt.output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function LSTM.ArraylstmSoftAtten(input_size, rnn_size, n, dropout)
  dropout = dropout or 0

  -- there will be 5*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    for S = 1,4 do
      table.insert(inputs, nn.Identity()()) -- prev_c_[S][L]. S is from 1 to 4
    end
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h   = inputs[(L-1)*5+6]
    local prev_c = {}
    for C = 1,4 do
      table.insert(prev_c, inputs[(L-1)*5+C+1])
    end
    -- the input to this layer
    if L == 1 then
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else
      x = outputs[(L-1)*5]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local attention = {}
    local n1 = {}
    local n2 = {}
    local n3 = {}
    local n4 = {}

    for C =1,4 do
      local i2h =  nn.Linear(input_size_L, 5 * rnn_size)(x):annotate{name='i2h_'..L}
      local h2h = nn.Linear(rnn_size, 5 * rnn_size)(prev_h):annotate{name='h2h_'..L}
      local all_input_sums = nn.CAddTable()({i2h, h2h})
      local reshaped = nn.Reshape(5, rnn_size)(all_input_sums)
      local n1_tmp, n2_tmp, n3_tmp, n4_tmp, n5_tmp = nn.SplitTable(2)(reshaped):split(5)
      table.insert(n1, nn.Sigmoid()(n1_tmp))
      table.insert(n2, nn.Sigmoid()(n2_tmp))
      table.insert(n3, nn.Sigmoid()(n3_tmp))
      table.insert(n4, nn.Tanh()(n4_tmp))
      -- attention signals
      table.insert(attention, nn.Sigmoid()(n5_tmp))
    end

    local attention_sum = nn.Exp()(attention[1])
    for C =2,4 do
      attention_sum = nn.CAddTable()({attention_sum, nn.Exp()(attention[C])})
    end

    local next_c = {}
    local next_h = {}
    for C =1,4 do
      local attention_norm = nn.CDivTable()({nn.Exp()(attention[C]), attention_sum})
      local in_gate = nn.CMulTable()({n1[C],attention_norm})
      local forget_gate = nn.CMulTable()({n2[C],attention_norm})
      local out_gate = nn.CMulTable()({n3[C],attention_norm})
      local in_transform = n4[C]
      -- perform the LSTM update
      --table.insert(next_c, nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c[C]}),nn.CMulTable()({in_gate, in_transform})}))
      table.insert(next_c, nn.CAddTable()({nn.CMulTable()({nn.AddConstant(1,true)(nn.MulConstant(-1,true)(forget_gate)), prev_c[C]}),nn.CMulTable()({in_gate, in_transform})}))
      -- gated cells form the output
      table.insert(next_h, nn.CMulTable()({out_gate, nn.Tanh()(next_c[C])}))
    end

    local next_h_sum = next_h[1]
    for C = 2,4 do
      next_h_sum = nn.CAddTable()({next_h_sum, next_h[C]})
    end

    for C = 1,4 do
      table.insert(outputs, next_c[C])
    end
    table.insert(outputs, next_h_sum)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end
--------------------------------------------
function LSTM.ArrayLSTM_SoftAtt(input_size, output_size, rnn_size, n, dropout)
	dropout = dropout or 0
	local cell_size = 4
	-- there will be 5*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- x
	for L = 1,n do
		for S = 1,cell_size do
			table.insert(inputs, nn.Identity()()) -- prev_c_[S][L]. S is from 1 to 4
		end
		table.insert(inputs, nn.Identity()()) -- prev_h[L]
	end

	local x, input_size_L
	local outputs = {}
	for L = 1,n do
		-- c,h from previos timesteps
		local prev_h   = inputs[(L-1)*(cell_size+1)+(cell_size+2)]
		local prev_c = {}
		for C = 1,cell_size do
			table.insert(prev_c, inputs[(L-1)*(cell_size+1)+C+1])
		end
		-- the input to this layer
		if L == 1 then
			x = inputs[1]
			-- x = OneHot(input_size)(inputs[1])
			input_size_L = input_size
		else
			x = outputs[(L-1)*(cell_size+1)]
			if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
			input_size_L = rnn_size
		end
		-- evaluate the input sums at once for efficiency
		local next_c = {}
		local next_h = {}

		for C =1,cell_size do
			local i2h = nn.Linear(input_size_L, (cell_size+1) * rnn_size)(x):annotate{name='i2h_'..L}
            local h2h = nn.Linear(rnn_size, (cell_size+1) * rnn_size)(prev_h):annotate{name='h2h_'..L}
            local all_input_sums = nn.CAddTable()({i2h, h2h})
            local reshaped = nn.Reshape((cell_size+1), rnn_size)(all_input_sums)
			local n1_tmp, n2_tmp, n3_tmp, n4_tmp, n5_tmp = nn.SplitTable(2)(reshaped):split(cell_size+1)
			local n1 = nn.Sigmoid()(n1_tmp)
			local n2 = nn.Sigmoid()(n2_tmp)
			local n3 = nn.Sigmoid()(n3_tmp)
			local n4 = nn.Tanh()(n4_tmp)
			local n5 = nn.Sigmoid()(n5_tmp)
			local weights = nn.SoftMax()(n5)
			local in_gate = nn.CMulTable(){n1,weights}
			local forget_gate = nn.CMulTable(){n2,weights}
			local out_gate = nn.CMulTable(){n3,weights}
			local in_transform = n4
			-- perform the LSTM update
			table.insert(next_c, nn.CAddTable()({nn.CMulTable()({nn.AddConstant(1,true)(nn.MulConstant(-1,true)(forget_gate)), prev_c[C]}),nn.CMulTable()({in_gate, in_transform})}))
			-- gated cells form the output
			table.insert(next_h, nn.CMulTable()({out_gate, nn.Tanh()(next_c[C])}))
		end

		local next_h_sum = next_h[1]
		for C = 2,cell_size do
			next_h_sum = nn.CAddTable()({next_h_sum, next_h[C]})
		end

		for C = 1,cell_size do
			table.insert(outputs, next_c[C])
		end
		table.insert(outputs, next_h_sum)
	end

	-- set up the decoder
	local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
	local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

	return nn.gModule(inputs, outputs)
end
--------------------------------------------
function LSTM.ArraylstmStochasticPooling(input_size, rnn_size, n, dropout)
  dropout = dropout or 0

  -- there will be 5*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    for S = 1,4 do
      table.insert(inputs, nn.Identity()()) -- prev_c_[S][L]. S is from 1 to 4
    end
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h   = inputs[(L-1)*5+6]
    local prev_c = {}
    for C = 1,4 do
      table.insert(prev_c, inputs[(L-1)*5+C+1])
    end
    -- the input to this layer
    if L == 1 then
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else
      x = outputs[(L-1)*5]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local n1 = {}
    local n2 = {}
    local n3 = {}
    local n4 = {}

    for C =1,4 do
      local i2h =  nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
      local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
      local all_input_sums = nn.CAddTable()({i2h, h2h})
      local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
      local n1_tmp, n2_tmp, n3_tmp, n4_tmp = nn.SplitTable(2)(reshaped):split(4)
      -- only has 4
      table.insert(n1, nn.Sigmoid()(n1_tmp))
      table.insert(n2, nn.Sigmoid()(n2_tmp))
      table.insert(n3, nn.Sigmoid()(n3_tmp))
      table.insert(n4, nn.Tanh()(n4_tmp))
    end

    local output_gates = nn.ConcatTable()({nn.ConcatTable()({n2[1],n2[2]}),nn.ConcatTable()({n2[3],n2[4]})})
    local output_proj = nn.Linear(4 * rnn_size, 4)(n2):annotate{name='stochasticdecoder'}
    local output_gates_soft = nn.Sigmoid()(nn.LogSoftMax()(output_proj))

    local next_c = {}
    local next_h = {}
    for C =1,4 do
      -- perform the LSTM update
      local out_gate = n2[C]
      local in_gate = n1[C]
      local forget_gate =n3[C]
      local in_transform = n4[C]
      table.insert(next_c, nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c[C]}),nn.CMulTable()({in_gate, in_transform})}))
      -- gated cells form the output
      table.insert(next_h, nn.CMulTable()({out_gate, nn.Tanh()(next_c[C])}))
    end

    local next_h_sum = nn.CMulTable()({next_h[1],output_gates_soft[1]})
    for C = 2,4 do
      next_h_sum = nn.CAddTable()({next_h_sum, nn.CMul(output_gates_soft[C])(next_h[C])})
    end
    for C = 1,4 do
      table.insert(outputs, next_c[C])
    end
    table.insert(outputs, next_h_sum)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function LSTM.SEQLSTM_DNN(input_size, output_size, rnn_size, n, dropout, bn)
	blstm = nn.Sequential()
	local lstm = cudnn.BLSTM(input_size, rnn_size, n)
	if dropout > 0 then
		lstm.dropout = dropout
		lstm.seed = 123
		lstm:resetDropoutDescriptor()
	end

	-- create softmax
	local softmax = nn.Sequential()
	if dropout > 0 then
		softmax:add( nn.Dropout(dropout, false, true) )
	end
	softmax:add( nn.Linear(rnn_size * 2, output_size) )
	softmax:add( nn.LogSoftMax() )

	local m = nn.Parallel()
	m:add(lstm)
	m:add(softmax)

	return {m=m, core=lstm, softmax=softmax}
end

function LSTM.SCLSTM2(input_size, output_size, rnn_size, n, dropout)
	dropout = dropout or 0

	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
	for L = 1,n do
		table.insert(inputs, nn.Identity()()) -- prev_c[L]
		table.insert(inputs, nn.Identity()()) -- prev_h[L]
	end

	local x, input_size_L
	local outputs = {}
	local pev_hidden = {}
	for L = 1,n do
		-- c,h from previos timesteps
		local prev_h = inputs[L*2+1]
		local prev_c = inputs[L*2]

		-- the input to this layer
		if L == 1 then
			x = inputs[1]
			input_size_L = input_size
		else
			x = outputs[(L-1)*2]
			if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
			input_size_L = rnn_size
		end
		-- evaluate the input sums at once for efficiency
		local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
		local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
		local all_input_sums = nn.CAddTable()({i2h, h2h})

		local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
		local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
		-- decode the gates
		local in_gate = nn.Sigmoid()(n1)
		local forget_gate = nn.Sigmoid()(n2)
		local out_gate = nn.Sigmoid()(n3)
		-- decode the write inputs
		local in_transform = nn.Tanh()(n4)
		-- perform the LSTM update
		local next_c           = nn.CAddTable()({
			nn.CMulTable()({forget_gate, prev_c}),
			nn.CMulTable()({in_gate,     in_transform})
		})
		-- gated cells form the output
		local next_h
		if L == 1 then
			next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
		else
			next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
			next_h = nn.CMulTable()({next_h, pev_hidden[L-1]})
		end
		table.insert(pev_hidden, next_h)
		table.insert(outputs, next_c)
		table.insert(outputs, next_h)
	end

	-- set up the decoder
	local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
	local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

	return nn.gModule(inputs, outputs)
end

function LSTM.SCLSTM(input_size, output_size, rnn_size, n, dropout)
	dropout = dropout or 0

	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
	table.insert(inputs, nn.Identity()()) -- sc_h[L]
	table.insert(inputs, nn.Identity()()) -- prev_c[L]
	table.insert(inputs, nn.Identity()()) -- prev_h[L]

	local x, input_size_L
	local outputs = {}
	local sc_h_in = inputs[2]
	-- c,h from previos timesteps
	local prev_c = inputs[3]
	local prev_h = inputs[4]
	-- the input to this layer
	x = inputs[1]
	input_size_L = input_size
	-- evaluate the input sums at once for efficiency
	local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..1}
	local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..1}
	local all_input_sums = nn.CAddTable()({i2h, h2h})

	local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
	local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
	-- decode the gates
	local in_gate = nn.Sigmoid()(n1)
	local forget_gate = nn.Sigmoid()(n2)
	local out_gate = nn.Sigmoid()(n3)
	-- decode the write inputs
	local in_transform = nn.Tanh()(n4)
	-- perform the LSTM update
	local next_c	= nn.CAddTable()({
					nn.CMulTable()({forget_gate, prev_c}),
					nn.CMulTable()({in_gate,     in_transform})
					})
	-- gated cells form the output
	local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
	local sc_h = nn.CAddTable()({next_h, sc_h_in})
	table.insert(outputs, next_c)
	table.insert(outputs, sc_h)

	-- set up the decoder
	local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
	local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

	return nn.gModule(inputs, outputs)
end

function LSTM.SEQLSTM(input_size, output_size, rnn_size, n, dropout, bn)
    dropout = dropout or 0

    -- there will be 2*n+1 inputs
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    for L = 1, n do
        table.insert(inputs, nn.Identity()()) -- prev_c[L]
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end

    local x, input_size_L
    local outputs = {}
    for L = 1, n do
        -- c,h from previos timesteps
        local prev_h = inputs[L * 2 + 1]
        local prev_c = inputs[L * 2]
        -- the input to this layer
        if L == 1 then
            x = inputs[1]
            input_size_L = input_size
        else
            x = outputs[(L - 1) * 2]
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size
        end
        -- recurrent batch normalization
        -- http://arxiv.org/abs/1603.09025
        local bn_wx, bn_wh, bn_c
        if bn then
            bn_wx = nn.BatchNormalization(4 * rnn_size, 1e-5, 0.1, true)
            bn_wh = nn.BatchNormalization(4 * rnn_size, 1e-5, 0.1, true)
            bn_c = nn.BatchNormalization(rnn_size, 1e-5, 0.1, true)

            -- initialise beta=0, gamma=0.1
            bn_wx.weight:fill(0.1)
            bn_wx.bias:zero()
            bn_wh.weight:fill(0.1)
            bn_wh.bias:zero()
            bn_c.weight:fill(0.1)
            bn_c.bias:zero()
        else
            bn_wx = nn.Identity()
            bn_wh = nn.Identity()
            bn_c = nn.Identity()
        end
        -- evaluate the input sums at once for efficiency
        local i2h = bn_wx(nn.Linear(input_size_L, 4 * rnn_size)(x):annotate { name = 'i2h_' .. L }):annotate { name = 'bn_wx_' .. L }
        local h2h = bn_wh(nn.Linear(rnn_size, 4 * rnn_size, false)(prev_h):annotate { name = 'h2h_' .. L }):annotate { name = 'bn_wh_' .. L }
        local all_input_sums = nn.CAddTable()({ i2h, h2h })

        local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
        -- decode the gates
        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local out_gate = nn.Sigmoid()(n3)
        -- decode the write inputs
        local in_transform = nn.Tanh()(n4)
        -- perform the LSTM update
        local next_c = nn.CAddTable()({
            nn.CMulTable()({ forget_gate, prev_c }),
            nn.CMulTable()({ in_gate, in_transform })
        })
        -- gated cells form the output
        local next_h = nn.CMulTable()({ out_gate, nn.Tanh()(bn_c(next_c):annotate { name = 'bn_c_' .. L }) })

        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end

	local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
	local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

    return nn.gModule(inputs, outputs)
end

function LSTM.SEQLSTM_INIT(n, des_lstm, src_lstm)

    for L = 1, n do
		for i, node_des in ipairs(des_lstm.forwardnodes) do
            if node_des.data.annotations.name == "i2h_" .. L then
				for _, node_src in ipairs(src_lstm.forwardnodes) do
					if node_src.data.annotations.name == "i2h_" .. L then
						des_lstm.forwardnodes[i].data.module.weight = node_src.data.module.weight
						des_lstm.forwardnodes[i].data.module.bias = node_src.data.module.bias
					end
				end
            end
        end

		for i, node_des in ipairs(des_lstm.forwardnodes) do
            if node_des.data.annotations.name == "h2h_" .. L then
				for _, node_src in ipairs(src_lstm.forwardnodes) do
					if node_src.data.annotations.name == "h2h_" .. L then
						des_lstm.forwardnodes[i].data.module.weight = node_src.data.module.weight
						des_lstm.forwardnodes[i].data.module.bias = node_src.data.module.bias
					end
				end
            end
        end

    end

	for i, node_des in ipairs(des_lstm.forwardnodes) do
		if node_des.data.annotations.name == "decoder" then
			for _, node_src in ipairs(src_lstm.forwardnodes) do
				if node_src.data.annotations.name == "decoder"  then
					des_lstm.forwardnodes[i].data.module.weight = node_src.data.module.weight
					des_lstm.forwardnodes[i].data.module.bias = node_src.data.module.bias
				end
			end
		end
	end
	return des_lstm
end



function LSTM.SEQLSTM_MM(input_size, output_size, rnn_size, n, dropout, bn)
    -- there will be 2*n+1 inputs
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    for L = 1, n do
        table.insert(inputs, nn.Identity()()) -- prev_c[L]
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end
	local reshaped_in1 = nn.Reshape(2, rnn_size)(inputs[1])
	local char_in, char_cnn = nn.SplitTable(2)(reshaped_in1):split(2)

    local x, input_size_L
    local outputs = {}
    for L = 1, n do
        -- c,h from previos timesteps
        local prev_h = inputs[L * 2 + 1]
        local prev_c = inputs[L * 2]
        -- the input to this layer
        if L == 1 then
            x = char_in
            input_size_L = input_size/2
        else
            x = outputs[(L - 1) * 2]
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size
        end
        -- recurrent batch normalization
        -- http://arxiv.org/abs/1603.09025
        local bn_wx, bn_wh, bn_c
        if bn then
            bn_wx = nn.BatchNormalization(4 * rnn_size, 1e-5, 0.1, true)
            bn_wh = nn.BatchNormalization(4 * rnn_size, 1e-5, 0.1, true)
            bn_c = nn.BatchNormalization(rnn_size, 1e-5, 0.1, true)

            -- initialise beta=0, gamma=0.1
            bn_wx.weight:fill(0.1)
            bn_wx.bias:zero()
            bn_wh.weight:fill(0.1)
            bn_wh.bias:zero()
            bn_c.weight:fill(0.1)
            bn_c.bias:zero()
        else
            bn_wx = nn.Identity()
            bn_wh = nn.Identity()
            bn_c = nn.Identity()
        end
        -- evaluate the input sums at once for efficiency
        local i2h = bn_wx(nn.Linear(input_size_L, 4 * rnn_size)(x):annotate { name = 'i2h_' .. L }):annotate { name = 'bn_wx_' .. L }
        local h2h = bn_wh(nn.Linear(rnn_size, 4 * rnn_size, false)(prev_h):annotate { name = 'h2h_' .. L }):annotate { name = 'bn_wh_' .. L }
        local all_input_sums = nn.CAddTable()({ i2h, h2h })

        local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
        -- decode the gates
        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local out_gate = nn.Sigmoid()(n3)
        -- decode the write inputs
        local in_transform = nn.Tanh()(n4)
        -- perform the LSTM update
        local next_c = nn.CAddTable()({
            nn.CMulTable()({ forget_gate, prev_c }),
            nn.CMulTable()({ in_gate, in_transform })
        })
        -- gated cells form the output
        local next_h = nn.CMulTable()({ out_gate, nn.Tanh()(bn_c(next_c):annotate { name = 'bn_c_' .. L }) })

        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end

	local top_h = outputs[#outputs]
	--[[
	local cat_outputs = {}
	table.insert(cat_outputs, top_h)
	table.insert(cat_outputs, char_cnn)
	local outconcat	= nn.JoinTable(2)(cat_outputs)
	]]
	local outconcat = nn.Tanh()(nn.CAddTable()({ nn.Linear(512,512)(top_h),nn.Linear(512,512)(char_cnn)}))
	local mm_map = nn.MulConstant(1.7159)(nn.Tanh()(outconcat))

	if dropout > 0 then mm_map = nn.Dropout(dropout)(mm_map):annotate{name='drop_final'} end
	local proj = nn.Linear(rnn_size, output_size)(mm_map):annotate{name='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

    return nn.gModule(inputs, outputs)
end


function LSTM.SEQLSTM_TORCH(input_size, output_size, rnn_size, n, dropout, bn)
    dropout = dropout or 0

    -- there will be 2*n+1 inputs
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    for L = 1, n do
        table.insert(inputs, nn.Identity()()) -- prev_c[L]
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end

    local x, input_size_L
    local outputs = {}
    for L = 1, n do
        -- c,h from previos timesteps
        local prev_h = inputs[L * 2 + 1]
        local prev_c = inputs[L * 2]
        -- the input to this layer
        if L == 1 then
            x = inputs[1]
            input_size_L = input_size
        else
            x = outputs[(L - 1) * 2]
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size
        end
        -- recurrent batch normalization
        -- http://arxiv.org/abs/1603.09025
        local bn_wx, bn_wh, bn_c
        if bn then
            bn_wx = nn.BatchNormalization(4 * rnn_size, 1e-5, 0.1, true)
            bn_wh = nn.BatchNormalization(4 * rnn_size, 1e-5, 0.1, true)
            bn_c = nn.BatchNormalization(rnn_size, 1e-5, 0.1, true)

            -- initialise beta=0, gamma=0.1
            bn_wx.weight:fill(0.1)
            bn_wx.bias:zero()
            bn_wh.weight:fill(0.1)
            bn_wh.bias:zero()
            bn_c.weight:fill(0.1)
            bn_c.bias:zero()
        else
            bn_wx = nn.Identity()
            bn_wh = nn.Identity()
            bn_c = nn.Identity()
        end
        -- evaluate the input sums at once for efficiency
        local i2h = bn_wx(nn.Linear(input_size_L, 4 * rnn_size)(x):annotate { name = 'i2h_' .. L }):annotate { name = 'bn_wx_' .. L }
        local h2h = bn_wh(nn.Linear(rnn_size, 4 * rnn_size, false)(prev_h):annotate { name = 'h2h_' .. L }):annotate { name = 'bn_wh_' .. L }
        local all_input_sums = nn.CAddTable()({ i2h, h2h })

        local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
        -- decode the gates
        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local out_gate = nn.Sigmoid()(n3)
        -- decode the write inputs
        local in_transform = nn.Tanh()(n4)
        -- perform the LSTM update
        local next_c = nn.CAddTable()({
            nn.CMulTable()({ forget_gate, prev_c }),
            nn.CMulTable()({ in_gate, in_transform })
        })
        -- gated cells form the output
        local next_h = nn.CMulTable()({ out_gate, nn.Tanh()(bn_c(next_c):annotate { name = 'bn_c_' .. L }) })

        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end

	local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
	local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

    return nn.gModule(inputs, outputs)
end


return LSTM
