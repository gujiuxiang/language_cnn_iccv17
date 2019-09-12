require 'nn'
require 'nngraph'

local DrawEnc = {}

function DrawEnc.LSTM(input_size, output_size, rnn_size, n, dropout)
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
        local next_c = nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c}), nn.CMulTable()({in_gate,in_transform})})
        -- gated cells form the output
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

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

function DrawEnc.qsample(rnn_size, n_z)
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()())   -- encoder hidden state
	table.insert(inputs, nn.Identity()())   -- e
    local next_h = inputs[1]
    local e = inputs[2]
	-------------------- next draw state --------------------
	local mu = nn.Linear(rnn_size, n_z)(next_h) -- calcuate the mu with 'encoder hidden'
	local sigma = nn.Linear(rnn_size, n_z)(next_h) -- calcuate the sigma with 'encoder hidden'
	sigma = nn.Exp()(sigma) -- sigma
    -- sampe 'z' from drawn from the latent distribution is passed as input to the decoder.
	local sigma_e = nn.CMulTable()({sigma, e}) --
	local z = nn.CAddTable()({mu, sigma_e})
	-- calcuate 'loss_z'
    local mu_squared = nn.Square()(mu)
	local sigma_squared = nn.Square()(sigma)
	local log_sigma_sq = nn.Log()(sigma_squared)
	local minus_log_sigma = nn.MulConstant(-1)(log_sigma_sq)
	local loss_z = nn.CAddTable()({mu_squared, sigma_squared, minus_log_sigma})
	loss_z = nn.AddConstant(-1)(loss_z)
	loss_z = nn.MulConstant(0.5)(loss_z)
	loss_z = nn.Sum(2)(loss_z)

	table.insert(outputs, z)
	table.insert(outputs, loss_z)
	-- packs the graph into a convenient module with standard API (:forward(), :backward())
	-- return nn.gModule(inputs, outputs)
	return nn.gModule(inputs, outputs)
end

function DrawEnc.read(batch_size, rnn_size)
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()())   -- input image (28*28)
    table.insert(inputs, nn.Identity()())   -- error previous
    local x_raw =inputs[1]
	local x_error_prev = inputs[2]
	local input = nn.JoinTable(2)({x_raw, x_error_prev}) -- combine two tensor 
    local x = nn.Reshape(2*3*32*32)(input)
	local output = nn.Linear(2*3*32*32, rnn_size)(x)
    table.insert(outputs, output)
	return nn.gModule(inputs, outputs)
end

function DrawEnc.convE(output_size)
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()())   -- input image (3*64*64)

    local x = inputs[1]
    local x_error_prev = inputs[2]
    local input = nn.JoinTable(1)({x, x_error_prev}) -- combine two tensor 
    -- input is 3 x 64 x 64
    local conv1 = nn.SpatialConvolution(3, 196, 4, 4, 2, 2, 1, 1)(input) -- becaureful !!!
    conv1 = nn.LeakyReLU(0.2, true)(conv1)
    -- state size: 196 x 32 x 32
    local conv2 = nn.SpatialConvolution(196, 196 * 2, 4, 4, 2, 2, 1, 1)(conv1)
    local norm2 = nn.SpatialBatchNormalization(196 * 2)(conv2)
    conv2 = nn.LeakyReLU(0.2, true)(norm2)
    -- state size: (196*2) x 16 x 16
    local conv3 = nn.SpatialConvolution(196 * 2, 196 * 4, 4, 4, 2, 2, 1, 1)(conv2)
    local norm3 = nn.SpatialBatchNormalization(196 * 4)(conv3)
    conv3 = nn.LeakyReLU(0.2, true)(norm3)
    -- state size: (196*4) x 8 x 8
    local conv4 = nn.SpatialConvolution(196 * 4, 196 * 8, 4, 4, 2, 2, 1, 1)(conv3)
    conv4 = nn.SpatialBatchNormalization(196 * 8)(conv4)
    -- state size: (196*8) x 4 x 4
    local conv5 = nn.SpatialConvolution(196 * 8, 196 * 2, 1, 1, 1, 1, 0, 0)(conv4)
    local norm5 = nn.SpatialBatchNormalization(196 * 2)(conv5)
    conv5 = nn.LeakyReLU(0.2, true)(norm5)
    local conv6 = nn.SpatialConvolution(196 * 2, 196 * 2, 3, 3, 1, 1, 1, 1)(conv5)
    local norm6 = nn.SpatialBatchNormalization(196 * 2)(conv6)
    conv6 = nn.LeakyReLU(0.2, true)(norm6)
    local conv7 = nn.SpatialConvolution(196 * 2, 196 * 8, 3, 3, 1, 1, 1, 1)(conv6)
    local norm7 = nn.SpatialBatchNormalization(196 * 8)(conv7)
    conv7 = nn.LeakyReLU(0.2, true)(norm7)
    -- state size: (196*8 ) x 4 x 4
    local conv8 = nn.SpatialConvolution(196 * 8 , 196 * 8, 1, 1)(conv7)
    local norm8 = nn.SpatialBatchNormalization(196 * 8)(conv8)
    conv8 = nn.LeakyReLU(0.2, true)(norm8)
    -- output convE_output_size x 1 x 1 feature map
    local conv9 = nn.SpatialConvolution(196 * 8, output_size, 4, 4)(conv8)
    conv9 = nn.Sigmoid()(conv9)
    output = nn.View(1):setNumInputDims(3)(conv9) -- change it to 1 x convE_output_size feature map

    -- apply layer 'weights initalization'
    local function weights_init(m)
       local name = torch.type(m)
       if name:find('Convolution') then
          m.weight:normal(0.0, 0.02)
          m.bias:fill(0)
       elseif name:find('BatchNormalization') then
          if m.weight then m.weight:normal(1.0, 0.02) end
          if m.bias then m.bias:fill(0) end
       end
    end
    --convE:apply(layer:weights_init)
	table.insert(outputs, output)   -- input image (64*64)
    return  nn.gModule(inputs, outputs)
end

return DrawEnc
