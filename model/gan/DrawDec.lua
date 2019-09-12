require 'nn'
require 'nngraph'

local DrawDec = {}

function DrawDec.LSTM(input_size, output_size, rnn_size, n, dropout)
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


function DrawDec.write(rnn_size)
	local inputs = {}
	local outputs = {}
    table.insert(inputs, nn.Identity()())   -- input hidden state of decoder
    table.insert(inputs, nn.Identity()())   -- input raw_image
    table.insert(inputs, nn.Identity()())   -- input previsou canvas
   
    local next_h = inputs[1]
    local raw_image = inputs[2]
    local image = nn.Reshape(3*32*32)(raw_image)
    local prev_canvas = inputs[3] 
    local write_layer = nn.Linear(512, 3*32*32)(next_h)
	local next_canvas = nn.CAddTable()({prev_canvas, write_layer})
	local mu = nn.Sigmoid()(next_canvas)
	local neg_mu = nn.MulConstant(-1)(mu)
	local d = nn.CAddTable()({image, neg_mu})
	--local d2 = nn.Power(2)(d)
	--local img_loss = nn.Sum(2)(d2)
	--local img_prediction = nn.Reshape(28, 28)(mu)
	local img_error = d
	
	table.insert(outputs, next_canvas)
	--table.insert(outputs, img_prediction)
	table.insert(outputs, img_error)
	--table.insert(outputs, img_loss)
	-- packs the graph into a convenient module with standard API (:forward(), :backward())
		
	module_write = nn.gModule(inputs, outputs)
	
	return module_write
end

function DrawDec.convD(input_size)
	local inputs = {}
	local outputs = {}
    table.insert(inputs, nn.Identity()())   -- input hidden state of decoder

    local hidden_dec = inputs[1]    -- rnn_size
    local raw_image = inputs[2]     -- 32x32
    local prev_canvas = inputs[3]   -- 32x32 

    local input = hidden_dec
    -- input is Z, going into a convolution
    local deconv1 = nn.SpatialFullConvolution(input_size, 196 * 8, 4, 4)(input)
    deconv1 = nn.SpatialBatchNormalization(196 * 8)(deconv1)
    -- state size: (196*8) x 4 x 4
    local deconv2 = nn.SpatialConvolution(196 * 8, 196 * 2, 1, 1, 1, 1, 0, 0)(deconv1)
    local norm2   = nn.SpatialBatchNormalization(196 * 2)(deconv2)
    deconv2 = nn.ReLU(true)(norm2)
    local deconv3 = nn.SpatialConvolution(196 * 2, 196 * 2, 3, 3, 1, 1, 1, 1)(deconv2)
    local norm3   = nn.SpatialBatchNormalization(196 * 2)(deconv3)
    deconv3 = nn.ReLU(true)(norm3)
    local deconv4 = nn.SpatialConvolution(196 * 2, 196 * 8, 3, 3, 1, 1, 1, 1)(deconv3)
    local norm4   = nn.SpatialBatchNormalization(196 * 8)(deconv4)
    deconv4 = nn.ReLU(true)(norm4)
    -- state size: (196*8) x 4 x 4
    local deconv5 = nn.SpatialFullConvolution(196 * 8, 196 * 4, 4, 4, 2, 2, 1, 1)(deconv4)
    deconv5 = nn.SpatialBatchNormalization(196 * 4)(deconv5)
    -- state size: (196*4) x 8 x 8
    local deconv6 = nn.SpatialConvolution(196 * 4, 196, 1, 1, 1, 1, 0, 0)(deconv5)
    local norm6   = nn.SpatialBatchNormalization(196)(deconv6)
    deconv6 = nn.ReLU(true)(norm6)
    local deconv7 = nn.SpatialConvolution(196, 196, 3, 3, 1, 1, 1, 1)(deconv6)
    local norm7   = nn.SpatialBatchNormalization(196)(deconv7)
    deconv7 = nn.ReLU(true)(norm7)
    local deconv8 = nn.SpatialConvolution(196, 196 * 4, 3, 3, 1, 1, 1, 1)(deconv7)
    local norm8   = nn.SpatialBatchNormalization(196 * 4)(deconv8)
    deconv8 = nn.ReLU(true)(norm8)
    -- state size: (196*4) x 8 x 8
    local deconv9 = nn.SpatialFullConvolution(196 * 4, 196 * 2, 4, 4, 2, 2, 1, 1)(deconv8)
    local norm9   = nn.SpatialBatchNormalization(196 * 2)(deconv9)
    deconv9 = nn.ReLU(true)(norm9)
    -- state size: (196*2) x 16 x 16
    local norm10  = nn.SpatialBatchNormalization(196)(deconv9)
    norm10  = nn.ReLU(true)(norm10)
    -- state size: (196) x 32 x 32
    local deconv11= nn.SpatialFullConvolution(196, 3, 4, 4, 2, 2, 1, 1)(norm10)
    local output  = nn.Tanh()(deconv11)
    -- state size: (3) x 64 x 64


	table.insert(outputs, output)

	-- packs the graph into a convenient module with standard API (:forward(), :backward())
	return nn.gModule(inputs, outputs)
end

return DrawDec
