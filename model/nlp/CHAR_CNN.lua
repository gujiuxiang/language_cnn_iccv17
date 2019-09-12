-- Time-delayed Neural Network (i.e. 1-d CNN) with multiple filter widths
require 'cudnn'
local CHAR_CNN = {}


function CHAR_CNN.tdnn()
    -- length = length of sentences/words (zero padded to be of same length)
    -- input_size = embedding_size
    -- feature_maps = table of feature maps (for each kernel width)
    -- kernels = table of kernel wsidths
    local layer1_concat, char_cnn_output
    local input = nn.Identity()() --input is batch_size x length x input_size
	local feature_maps = {32,64,64,96,128,128} --1100 filters
	local kernels = {1,2,3,4,5,6}
	local length = 18
	local input_size = 512
	local output_size = 512
    local layer1 = {}
    for i = 1, #kernels do
		local reduced_l = length - kernels[i] + 1 
		local pool_layer
		-- Use CuDNN for temporal convolution. Fake the spatial convolution.
		local conv = cudnn.SpatialConvolution(1, feature_maps[i], input_size, kernels[i], 1, 1, 0)
		conv.name = 'conv_filter_' .. kernels[i] .. '_' .. feature_maps[i]
		local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
		pool_layer = nn.Squeeze()(cudnn.SpatialMaxPooling(1, reduced_l, 1, 1, 0, 0)(nn.Tanh()(conv_layer)))
		table.insert(layer1, pool_layer)
    end
    if #kernels > 1 then
		layer1_concat = nn.JoinTable(2)(layer1)
		output = layer1_concat
    else
        output = layer1[1]
    end
	
	drop_output = nn.Dropout(0.5)(output) 
	local logsoft = nn.LogSoftMax()(drop_output)
	
    return nn.gModule({input}, {logsoft})
end


function CHAR_CNN.tdnn_dynamicKpool()
    -- length = length of sentences/words (zero padded to be of same length)
    -- input_size = embedding_size
    -- feature_maps = table of feature maps (for each kernel width)
    -- kernels = table of kernel wsidths
	local feature_maps = {32,64,64,96,128,128} --1100 filters
	local kernels = {1,2,3,4,5,6}
	local length = 18
	local input_size = 512
	local output_size = 512
    local layer1_concat, char_cnn_output
    local input = nn.Identity()() --input is batch_size x length x input_size
    
    local layer1 = {}
    for i = 1, #kernels do
		local reduced_l = length - kernels[i] + 1 
		local pool_layer
		-- Use CuDNN for temporal convolution.
		-- Temporal conv. much slower
		local conv = cudnn.TemporalConvolution(input_size, feature_maps[i], kernels[i])
		local conv_layer = conv(input)
		conv.name = 'conv_filter_' .. kernels[i] .. '_' .. feature_maps[i]
		--pool_layer = nn.Max(2)(nn.Tanh()(conv_layer))    
		--pool_layer = nn.TemporalMaxPooling(reduced_l)(nn.Tanh()(conv_layer))
		pool_layer = nn.TemporalDynamicKMaxPooling(1)(nn.Tanh()(conv_layer))
		pool_layer = nn.Squeeze()(pool_layer)
		table.insert(layer1, pool_layer)
    end
    if #kernels > 1 then
		layer1_concat = nn.JoinTable(2)(layer1)
		output = layer1_concat
    else
        output = layer1[1]
    end
	local tmp_cnn_out = 0
	for i = 1, #feature_maps do
		tmp_cnn_out = tmp_cnn_out + feature_maps[i]
	end
    return nn.gModule({input}, {output})
end


function CHAR_CNN:CapEncoderEmb()
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- img
	local captions = inputs[1]--bs, 18, 512
	local x = nn.Transpose({2,3})(captions)--bs,512, 18
	-- 512 x alphasize
	local conv1 = cudnn.TemporalConvolution(18, 256, 7)(x)--nOutputFrame = (nInputFrame - kW) / dW + 1=
	local thresh1 = nn.Threshold()(conv1)
	local max1 = nn.TemporalMaxPooling(3, 3)(thresh1) --nOutputFrame = (nInputFrame - kW) / dW + 1
	local conv2 = cudnn.TemporalConvolution(256, 256, 7)(max1)-- 168 x 256
	local thresh2 = nn.Threshold()(conv2)
	local max2 = nn.TemporalMaxPooling(3, 3)(thresh2) -- 54 x 256 :nOutputFrame = (nInputFrame - kW) / dW + 1
	local conv3 = cudnn.TemporalConvolution(256, 256, 3)(max2)  
	local thresh3 = nn.Threshold()(conv3)
	local max3 = nn.TemporalMaxPooling(3, 3)(thresh3)-- 17 x 256
	local conv4 = cudnn.TemporalConvolution(256, 256, 3)(max3)
	local thresh4 = nn.Threshold()(conv4)
	local max4 = nn.TemporalMaxPooling(3, 3)(thresh4)-- 5 x 256
	local conv5 = cudnn.TemporalConvolution(256, 256, 3)(max4)
	local thresh5 = nn.Threshold()(conv5)
	
	local rs = nn.Reshape(768)(thresh5)
	local rs_output = nn.Dropout(0.5)(rs) 
	local fc2 = nn.Linear(768, 512)(rs_output)
	local logsoft = nn.LogSoftMax()(fc2)
	outputs = {}
	table.insert(outputs, logsoft)
	
	return nn.gModule(inputs, outputs)
end
--[[
use cnn to encode the input captions,
batch_size, 18, 512
]]
function CHAR_CNN.Caps_CNN_short()
	local net = nn.Sequential() --bs, 18, alphasize
	-- 18 x alphasize
	net:add(cudnn.TemporalConvolution(512, 256, 3))
	net:add(nn.Threshold())
	-- 16 x 256
	net:add(nn.TemporalMaxPooling(3, 3))
	-- 5 x 256
	net:add(cudnn.TemporalConvolution(256, 256, 3))
	net:add(nn.Threshold())
	-- 3 x 256
	net:add(nn.Reshape(768))
	net:add(nn.Linear(768, 512))
	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(512, 512))
	return net
end


return CHAR_CNN
