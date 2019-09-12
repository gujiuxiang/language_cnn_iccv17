require 'nngraph'
require 'nn'
local attention = {}
--[[
https://github.com/JamesChuanggg/san-torch/blob/master/misc/attention.lua
]]
function attention.san_atten()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local caption_feat = inputs[1]	-- [batch_size, 512]
  local img_feat = inputs[2]	-- [batch_size, 14*14, 512] , conv5 pool output feature,

  -- stack attention
  local caption_emb_1 = nn.Linear(512, 512)(caption_feat)   						-- [batch_size, 512]
  local caption_emb_expand_1 = nn.Replicate(49, 2)(caption_emb_1)				-- [batch_size, 14*14, att_size]
  local img_emb_dim_1 = nn.Linear(512, 512, false)(nn.View(-1, 512)(img_feat)) 	-- [batch_size*m, 512]
  local img_emb_1 = nn.View(-1, 49, 512)(img_emb_dim_1)							-- [batch_size, 196, 512]
  local h1 = nn.Tanh()(nn.CAddTable()({caption_emb_expand_1, img_emb_1})) 			-- calcuate the attention distribution over the regions
  local h1_drop = nn.Dropout(0.5)(h1)	                     	  			-- [batch_size, m, att_size]
  local h1_emb = nn.Linear(512, 1)(nn.View(-1,512)(h1_drop))  					-- [batch_size * 196, 1]
  local p1 = nn.SoftMax()(nn.View(-1,196)(h1_emb))       							-- [batch_size, 196]
  local p1_att = nn.View(1,-1):setNumInputDims(1)(p1)             				-- [batch_size, 1, 196]
  local img_Att1 = nn.MM(false, false)({p1_att, img_feat})	    				-- [batch_size, 1, 512] , equation 17 -- Weighted Sum
  local img_att_feat_1 = nn.View(-1, 512)(img_Att1)	    						-- [batch_size, 512]
  local u1 = nn.CAddTable()({caption_feat, img_att_feat_1})	    					-- [batch_size, 512] , equation 18

  -- Stack 2
  local caption_emb_2 = nn.Linear(512, 512)(u1)          -- [batch_size, 512]
  local caption_emb_expand_2 = nn.Replicate(196,2)(caption_emb_2) 	    -- [batch_size, m, 512]
  local img_emb_dim_2 = nn.Linear(512, 512, false)(nn.View(-1,512)(img_feat)) -- [batch_size*m, 512]
  local img_emb_2 = nn.View(-1, 196, 512)(img_emb_dim_2)			          -- [batch_size, m, 512]
  local h2 = nn.Tanh()(nn.CAddTable()({caption_emb_expand_2, img_emb_2}))
  local h2_drop = nn.Dropout(0.5)(h2)    	          -- [batch_size, m, 512]
  local h2_emb = nn.Linear(512, 1)(nn.View(-1,512)(h2_drop)) -- [batch_size * m, 1]
  local p2 = nn.SoftMax()(nn.View(-1,196)(h2_emb))       -- [batch_size, m]
  local p2_att = nn.View(1,-1):setNumInputDims(1)(p2)             -- [batch_size, 1, m]
  local img_Att2 = nn.MM(false, false)({p2_att, img_feat})        -- [batch_size, 1, d]
  local img_att_feat_2 = nn.View(-1, 512)(img_Att2)        -- [batch_size, d]
  local u2 = nn.CAddTable()({u1, img_att_feat_2})		                  -- [batch_size, d]

  table.insert(outputs, u2)

  return nn.gModule(inputs, outputs)
end

function attention.san_atten_196()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())


  local img_feat = nn.View(-1, 196, 512)(inputs[1])	-- [batch_size, 14*14, 512] , conv5 pool output feature,
  local ques_feat = inputs[2]	-- [batch_size, 512]
  local input_size = 512
  local att_size = 512
  local img_seq_size = 196
  local output_size = 512
  local drop_ratio = 0.5

  -- stack attention
  local ques_emb_1 = nn.Linear(input_size, att_size)(ques_feat)   -- [batch_size, att_size]
  local ques_emb_expand_1 = nn.Replicate(img_seq_size,2)(ques_emb_1)         -- [batch_size, m, att_size]
  local img_emb_dim_1 = nn.Linear(input_size, att_size, false)(nn.View(-1,input_size)(img_feat)) -- [batch_size*m, att_size]
  local img_emb_1 = nn.View(-1, img_seq_size, att_size)(img_emb_dim_1)        		          -- [batch_size, m, att_size]
  local h1 = nn.Tanh()(nn.CAddTable()({ques_emb_expand_1, img_emb_1}))
  local h1_drop = nn.Dropout(drop_ratio)(h1)	                     	  -- [batch_size, m, att_size]
  local h1_emb = nn.Linear(att_size, 1)(nn.View(-1,att_size)(h1_drop))  -- [batch_size * m, 1]
  local p1 = nn.SoftMax()(nn.View(-1,img_seq_size)(h1_emb))       -- [batch_size, m]
  local p1_att = nn.View(1,-1):setNumInputDims(1)(p1)             -- [batch_size, 1, m]
  -- Weighted Sum
  local img_Att1 = nn.MM(false, false)({p1_att, img_feat})	    -- [batch_size, 1, d]
  local img_att_feat_1 = nn.View(-1, input_size)(img_Att1)	    -- [batch_size, d]
  local u1 = nn.CAddTable()({ques_feat, img_att_feat_1})	    -- [batch_size, d]


  -- Stack 2
  local ques_emb_2 = nn.Linear(input_size, att_size)(u1)          -- [batch_size, att_size]
  local ques_emb_expand_2 = nn.Replicate(img_seq_size,2)(ques_emb_2) 	    -- [batch_size, m, att_size]
  local img_emb_dim_2 = nn.Linear(input_size, att_size, false)(nn.View(-1,input_size)(img_feat)) -- [batch_size*m, att_size]
  local img_emb_2 = nn.View(-1, img_seq_size, att_size)(img_emb_dim_2)			          -- [batch_size, m, att_size]
  local h2 = nn.Tanh()(nn.CAddTable()({ques_emb_expand_2, img_emb_2}))
  local h2_drop = nn.Dropout(drop_ratio)(h2)    	          -- [batch_size, m, att_size]
  local h2_emb = nn.Linear(att_size, 1)(nn.View(-1,att_size)(h2_drop)) -- [batch_size * m, 1]
  local p2 = nn.SoftMax()(nn.View(-1,img_seq_size)(h2_emb))       -- [batch_size, m]
  local p2_att = nn.View(1,-1):setNumInputDims(1)(p2)             -- [batch_size, 1, m]
  -- Weighted Sum
  local img_Att2 = nn.MM(false, false)({p2_att, img_feat})        -- [batch_size, 1, d]
  local img_att_feat_2 = nn.View(-1, input_size)(img_Att2)        -- [batch_size, d]
  local u2 = nn.CAddTable()({u1, img_att_feat_2})		                  -- [batch_size, d]

  table.insert(outputs, u2)
  return nn.gModule(inputs, outputs)
end

function attention.att_hidden_deconv()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())  -- tmax, bs, 512
  table.insert(inputs, nn.Identity()())  -- tmax, bs, 9568

  local hidden = inputs[1]
  local probs = inputs[2]
  local probs_trans = nn.Transpose({1,2})(probs) -- bs, tmax, 9568
  local hidden_trans = nn.Transpose({1,2})(hidden) -- bs, tmax, 512

  local probs_trans_1 = nn.View(-1, 9568)(probs_trans) -- bs*tmax, 512
  local probs_trans_1_emb = nn.Linear(9568, 512)(probs_trans_1)-- bs*tmax, 512
  local probs_trans_1_emb_p1 = nn.Linear(512, 1)(probs_trans_1_emb)-- bs*tmax, 1
  local probs_trans_1_emb_p2 = nn.SoftMax()(probs_trans_1_emb_p1)  --bs*tmax,1
  local probs_trans_att = nn.View(-1,1,18)(probs_trans_1_emb_p2) --bs, 1, tmax
  local im_out = nn.View(-1,512)(nn.MM(false, false)({probs_trans_att, hidden_trans})) --bs,512

  table.insert(outputs, im_out)

  return nn.gModule(inputs, outputs)
end

function attention.att_outemb_deconv()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())  -- tmax, bs, 512
  table.insert(inputs, nn.Identity()())  -- tmax, bs, 9568

  local hidden = inputs[1]
  local probs = inputs[2]
  local probs_trans = nn.Transpose({1,2})(probs) -- bs, tmax, 9568
  local hidden_trans = nn.Transpose({1,2})(hidden) -- bs, tmax, 512

  local probs_trans_1 = nn.View(-1, 512)(probs_trans) -- bs*tmax, 512
  local probs_trans_1_emb = nn.Linear(512, 512)(probs_trans_1)-- bs*tmax, 512
  local probs_trans_1_emb_p1 = nn.Linear(512, 1)(probs_trans_1_emb)-- bs*tmax, 1
  local probs_trans_1_emb_p2 = nn.SoftMax()(probs_trans_1_emb_p1)  --bs*tmax,1
  local probs_trans_att = nn.View(-1,1,18)(probs_trans_1_emb_p2) --bs, 1, tmax
  local im_out = nn.View(-1,512)(nn.MM(false, false)({probs_trans_att, hidden_trans})) --bs,512

  table.insert(outputs, im_out)

  return nn.gModule(inputs, outputs)
end

function attention.att_outemb_deconv_conv5_poolout()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())  -- tmax, bs, 512
  table.insert(inputs, nn.Identity()())  -- tmax, bs, 9568

  local hidden = inputs[1]
  local probs = inputs[2]
  local probs_trans = nn.Transpose({1,2})(probs) -- bs, tmax, 9568
  local hidden_trans = nn.Transpose({1,2})(hidden) -- bs, tmax, 512

  local probs_trans_1 = nn.View(-1, 512)(probs_trans) -- bs*tmax, 512
  local probs_trans_1_emb = nn.Linear(512, 512)(probs_trans_1)-- bs*tmax, 512
  local probs_trans_1_emb_p1 = nn.Linear(512, 1)(probs_trans_1_emb)-- bs*tmax, 1
  local probs_trans_1_emb_p2 = nn.SoftMax()(probs_trans_1_emb_p1)  --bs*tmax,1
  local probs_trans_att = nn.View(-1,1,18)(probs_trans_1_emb_p2) --bs, 1, tmax
  local im_out = nn.View(-1,512)(nn.MM(false, false)({probs_trans_att, hidden_trans})) --bs,512

  table.insert(outputs, im_out)

  return nn.gModule(inputs, outputs)
end

function attention.san_atten_4096()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local caption_feat = inputs[1]	-- [batch_size, 4096]
  local img_feat = inputs[2]	-- [batch_size, 4096] , conv5 pool output feature,

  -- stack attention
  local caption_emb_1 = nn.Tanh()(nn.Linear(4096, 4096)(caption_feat))	-- [batch_size, 4096]
  local caption_emb_2 = nn.Linear(4096, 4096)(caption_feat)	-- [batch_size, 4096]

  local img_emb_1 = nn.Tanh()(nn.Linear(4096, 4096)(img_feat))	-- [batch_size, 4096]
  local img_emb_2 = nn.Tanh()(nn.Linear(4096, 4096)(img_emb_1))
  local img_Att1 = nn.MM(false, false)({caption_emb_1, img_emb_2})
  -- [batch_size, 196, 512]
  local emb_out = nn.Tanh()(nn.CAddTable()({caption_emb_2, img_Att1}))
  table.insert(outputs, emb_out)

  return nn.gModule(inputs, outputs)

end

function attention:conv_att_san_49()
  -- T=196, eInputSize=512, dInputSize = 512
  local T = 49
  local eInputSize = 512
  local dInputSize = 512
  local function efficient1x1Conv(inputSize, outputDim, T)
    local module = nn.Sequential()
    module:add(nn.View(-1, inputSize))
    module:add(nn.Linear(inputSize, outputDim))
    if outputDim == 1 then
      module:add(nn.View(-1, T))
    else
      module:add(nn.View(-1, T, outputDim))
    end
    return module
  end

  local conv_h = nn.Identity()() --h is 3 dimensional: batch x T x eInputSize
  local dt = nn.Identity()() --dt is 2 dimensional: batch x dInputSize
  local h = nn.View(-1, 49, 512)(conv_h):annotate{name='view_1_196_512'} --[batch_size,14*14, 512]
  local W2dt = nn.Linear(dInputSize, eInputSize)(dt):annotate{name='MLP1'}
  local W2dtRepeated = nn.Replicate(T, 2, 2)(W2dt)
  local W1h = efficient1x1Conv(eInputSize, eInputSize, T)(h):annotate{name='efficient1x1Conv_W1h'}--nn.TemporalConvolution(eInputSize, eInputSize, 1, 1)(h)
  local TanSum = nn.Tanh()(nn.CAddTable()({W1h, W2dtRepeated})) -- size: batch x T x eInputSize
  local u = efficient1x1Conv(eInputSize, 1, T)(TanSum):annotate{name='efficient1x1Conv_u'} -- size: batch x T
  local a = nn.SoftMax()(u):annotate{name='SoftMax'}-- size: batch x T
  local weightedH = nn.CMulTable(){h, nn.Replicate(eInputSize, 2, 1)(a)}:annotate{name='CMulTable'}
  local dtN = nn.Sum(1, 2)(weightedH) --output size: batch x eInputSize
  --return nn.gModule({h,dt}, {dtN, a})
  return nn.gModule({conv_h,dt}, {dtN})
end

function attention.san_atten_512()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local caption_feat = inputs[1]	-- [batch_size, 4096]
  local img_feat = inputs[2]	-- [batch_size, 4096] , conv5 pool output feature,

  -- stack attention
  local caption_emb_1 = nn.Tanh()(nn.Linear(512, 512)(caption_feat))	-- [batch_size, 4096]
  local caption_emb_2 = nn.Linear(512, 512)(caption_feat)	-- [batch_size, 4096]

  local img_emb_1 = nn.Tanh()(nn.Linear(512, 512)(img_feat))	-- [batch_size, 4096]
  local img_emb_2 = nn.Tanh()(nn.Linear(512, 512)(img_emb_1))
  local img_Att1 = nn.CMulTable()({caption_emb_1, img_emb_2})
  local emb_out = nn.CAddTable()({caption_emb_2, img_Att1})
  table.insert(outputs, emb_out)

  return nn.gModule(inputs, outputs)
end

function attention.san_atten_deconv()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local caption_feat = inputs[1]	-- [18 ,batch_size,  9568], caption_feat->caption_hidden_feat
  local img_feat = inputs[2]	-- [batch_size, 512] , conv5 pool output feature,

  -- stack attention
  local img_emb_1 = nn.Linear(512, 512)(img_feat)   									-- [batch_size, 512]
  local img_emb_expand_1 = nn.Replicate(18, 2)(img_emb_1)								-- [batch_size, 18, att_size]
  local caption_feat_trans = nn.Transpose({1,2})(caption_feat)						-- [batch_size, 18 ,  9568]
  local caption_emb_dim_1 = nn.Linear(9568, 512, false)(nn.View(-1, 9568)(caption_feat_trans))-- [batch_size*18, 512]
  local caption_emb_1 = nn.View(-1, 18, 512)(caption_emb_dim_1)						-- [batch_size, 18, 512]
  local h1 = nn.Tanh()(nn.CAddTable()({img_emb_expand_1, caption_emb_1})) 			-- calcuate the attention distribution over the regions
  local h1_drop = nn.Dropout(0.5)(h1)	                     	  						-- [batch_size, m, att_size]
  local h1_emb = nn.Linear(512, 1)(nn.View(-1,512)(h1_drop))  						-- [batch_size * 18, 1]
  local p1 = nn.SoftMax()(nn.View(-1,18)(h1_emb))       								-- [batch_size, 18]
  local p1_att = nn.View(1,-1):setNumInputDims(1)(p1)             					-- [batch_size, 1, 18]
  local caption_Att1 = nn.MM(false, false)({p1_att, caption_emb_1})	    			-- [batch_size, 1, 512] , equation 17 -- Weighted Sum
  local caption_att_feat_1 = nn.View(-1, 512)(caption_Att1)	    					-- [batch_size, 512]
  local u1 = nn.CAddTable()({img_feat, caption_att_feat_1})	    					-- [batch_size, 512] , equation 18

  --table.insert(outputs, u1)
  local de_fc8 = nn.Linear(512, 4096)(u1)
  --table.insert(outputs, de_fc8)
  local de_relu8 = nn.ReLU(true)(de_fc8)
  local de_drop7 = nn.Dropout(0.500000)(de_relu8)
  local de_fc7 = nn.Linear(4096, 4096)(de_drop7)
  --table.insert(outputs, de_fc7)
  local de_drop6 = nn.Dropout(0.500000)(de_fc7)
  local de_relu6 = nn.ReLU(true)(de_drop6)
  local de_fc6 = nn.Linear(4096, 25088)(de_relu6)
  local de_output = nn.ReLU(true)(de_fc6)
  table.insert(outputs, de_output)

  return nn.gModule(inputs, outputs)
end

--[[
https://github.com/eladhoffer/recurrent.torch/blob/master/Attention.lua
]]
function attention:conv_att_san()
  -- T=196, eInputSize=512, dInputSize = 512
  local T = 196
  local eInputSize = 512
  local dInputSize = 512
  local function efficient1x1Conv(inputSize, outputDim, T)
    local module = nn.Sequential()
    module:add(nn.View(-1, inputSize))
    module:add(nn.Linear(inputSize, outputDim))
    if outputDim == 1 then
      module:add(nn.View(-1, T))
    else
      module:add(nn.View(-1, T, outputDim))
    end
    return module
  end

  local conv_h = nn.Identity()() --h is 3 dimensional: batch x T x eInputSize
  local dt = nn.Identity()() --dt is 2 dimensional: batch x dInputSize
  local h = nn.View(-1, 196, 512)(conv_h):annotate{name='view_1_196_512'} --[batch_size,14*14, 512]
  local W2dt = nn.Linear(dInputSize, eInputSize)(dt):annotate{name='MLP1'}
  local W2dtRepeated = nn.Replicate(T, 2, 2)(W2dt)
  local W1h = efficient1x1Conv(eInputSize, eInputSize, T)(h):annotate{name='efficient1x1Conv_W1h'}--nn.TemporalConvolution(eInputSize, eInputSize, 1, 1)(h)
  local TanSum = nn.Tanh()(nn.CAddTable()({W1h, W2dtRepeated})) -- size: batch x T x eInputSize
  local u = efficient1x1Conv(eInputSize, 1, T)(TanSum):annotate{name='efficient1x1Conv_u'} -- size: batch x T
  local a = nn.SoftMax()(u):annotate{name='SoftMax'}-- size: batch x T
  local weightedH = nn.CMulTable(){h, nn.Replicate(eInputSize, 2, 1)(a)}:annotate{name='CMulTable'}
  local dtN = nn.Sum(1, 2)(weightedH) --output size: batch x eInputSize
  --return nn.gModule({h,dt}, {dtN, a})
  return nn.gModule({conv_h,dt}, {dtN})
end

function attention:conv_att_san_2()
  -- T=196, eInputSize=512, dInputSize = 512
  local T = 196
  local eInputSize = 512
  local dInputSize = 512
  local function efficient1x1Conv(inputSize, outputDim, T)
    local module = nn.Sequential()
    module:add(nn.View(-1, inputSize))
    module:add(nn.Linear(inputSize, outputDim))
    module:add(cudnn.ReLU(true))
    if outputDim == 1 then
      module:add(nn.View(-1, T))
    else
      module:add(nn.View(-1, T, outputDim))
    end
    return module
  end

  local conv_h = nn.Identity()() --h is 3 dimensional: batch x T x eInputSize
  local dt = nn.Identity()() --dt is 2 dimensional: batch x dInputSize
  local h = nn.View(-1, 196, 512)(conv_h):annotate{name='view_1_196_512'} --[batch_size,14*14, 512]
  local W2dt = nn.Linear(dInputSize, eInputSize)(dt):annotate{name='MLP1'}
  local W2dtRepeated = nn.Replicate(T, 2, 2)(W2dt)
  local W1h = efficient1x1Conv(eInputSize, eInputSize, T)(h):annotate{name='efficient1x1Conv_W1h'}--nn.TemporalConvolution(eInputSize, eInputSize, 1, 1)(h)
  local TanSum = nn.Tanh()(nn.CAddTable()({W1h, W2dtRepeated})) -- size: batch x T x eInputSize
  local u = efficient1x1Conv(eInputSize, 1, T)(TanSum):annotate{name='efficient1x1Conv_u'} -- size: batch x T
  local a = nn.SoftMax()(u):annotate{name='SoftMax'}-- size: batch x T
  local weightedH = nn.CMulTable(){h, nn.Replicate(eInputSize, 2, 1)(a)}:annotate{name='CMulTable'}
  local dtN = nn.Sum(1, 2)(weightedH) --output size: batch x eInputSize

  --return nn.gModule({h,dt}, {dtN, a})
  return nn.gModule({conv_h,dt}, {dtN})
end

function attention:conv_att_san_3()
  -- T=196, eInputSize=512, dInputSize = 512
  local T = 196
  local eInputSize = 512
  local dInputSize = 512
  local function efficient1x1Conv(inputSize, outputDim, T)
    local module = nn.Sequential()
    module:add(nn.View(-1, inputSize))
    module:add(nn.Linear(inputSize, outputDim))
    module:add(cudnn.ReLU(true))
    if outputDim == 1 then
      module:add(nn.View(-1, T))
    else
      module:add(nn.View(-1, T, outputDim))
    end
    return module
  end

  local conv_h = nn.Identity()() --h is 3 dimensional: batch x T x eInputSize
  local dt = nn.Identity()() --dt is 2 dimensional: batch x dInputSize
  local h = nn.View(-1, 196, 512)(conv_h):annotate{name='view_1_196_512'} --[batch_size,14*14, 512]
  local W2dt = nn.Linear(dInputSize, eInputSize)(dt):annotate{name='MLP1'}
  local W2dtRepeated = nn.Replicate(T, 2, 2)(W2dt)
  local W1h = efficient1x1Conv(eInputSize, eInputSize, T)(h):annotate{name='efficient1x1Conv_W1h'}--nn.TemporalConvolution(eInputSize, eInputSize, 1, 1)(h)
  local TanSum = nn.Sigmoid()(nn.CAddTable()({W1h, W2dtRepeated})) -- size: batch x T x eInputSize
  local u = efficient1x1Conv(eInputSize, 1, T)(TanSum):annotate{name='efficient1x1Conv_u'} -- size: batch x T
  local a = nn.SoftMax()(u):annotate{name='SoftMax'}-- size: batch x T
  local weightedH = nn.CMulTable(){h, nn.Replicate(eInputSize, 2, 1)(a)}:annotate{name='CMulTable'}
  local dtN = nn.Sum(1, 2)(weightedH) --output size: batch x eInputSize

  --return nn.gModule({h,dt}, {dtN, a})
  return nn.gModule({conv_h,dt}, {dtN})
end

function attention.conv_att()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local hidden = inputs[2] --bs*512
  local conv5_ft = inputs[1]	-- [batch_size*14*14, 512] , conv5 pool output feature,

  --local conv5_ft1 = nn.Linear(512, 512)(nn.View(-1, 512)(conv5_ft)):annotate{name='MLP1'}
  local conv5_ft2 = nn.View(-1, 196, 512)(conv5_ft) --[batch_size,14*14, 512]
  --512-dimensional feature mask vector
  local hidden_mask = nn.Linear(512, 512)(hidden):annotate{name='MLP2'}
  local hidden_mask1 = nn.Sigmoid()(hidden_mask) -- mask_t
  local hidden_mask2 = nn.View(-1, 512, 1)(hidden_mask1) -- bs*512*1
  --probability distribution
  local p1 = nn.View(-1, 196)(nn.MM(false, false)({conv5_ft2, hidden_mask2})):annotate{name='MM1'} --bs*196*1
  --local p1_att = nn.SoftMax()(nn.Sigmoid()(p1)) --
  local p1_att = nn.SoftMax()(p1):annotate{name='SoftMax'} --
  local p1_att1 = nn.View(-1, 1, 196)(p1_att) --bs*1*196
  local img_Att2 = nn.MM(false, false)({p1_att1, conv5_ft2}):annotate{name='MM2'} --bs*1*512
  local img_Att3 = nn.View(-1, 512)(img_Att2)	    						-- [batch_size, 512]
  local img_Att4 = nn.Linear(512, 512)(img_Att3):annotate{name='MLP3'}

  table.insert(outputs, img_Att4)
  return nn.gModule(inputs, outputs)
end

function attention.conv_att_best()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local hidden = inputs[2] --bs*512
  local conv5_ft = inputs[1]	-- [batch_size*14*14, 512] , conv5 pool output feature,

  local conv5_ft1 = nn.Linear(512, 512)(nn.View(-1, 512)(conv5_ft)):annotate{name='MLP1'}
  local conv5_ft2 = nn.View(-1, 196, 512)(conv5_ft1) --[batch_size,14*14, 512]
  --512-dimensional feature mask vector
  local hidden_mask = nn.Linear(512, 512)(hidden):annotate{name='MLP2'}
  local hidden_mask1 = nn.Sigmoid()(hidden_mask) -- mask_t
  local hidden_mask2 = nn.View(-1, 512, 1)(hidden_mask1) -- bs*512*1
  --probability distribution
  local p1 = nn.View(-1, 196)(nn.MM(false, false)({conv5_ft2, hidden_mask2})):annotate{name='MM1'} --bs*196*1
  --local p1_att = nn.SoftMax()(nn.Sigmoid()(p1)) --
  local p1_att = nn.SoftMax()(p1):annotate{name='SoftMax'} --
  local p1_att1 = nn.View(-1, 1, 196)(p1_att) --bs*1*196
  local img_Att2 = nn.MM(false, false)({p1_att1, conv5_ft2}):annotate{name='MM2'} --bs*1*512
  local img_Att3 = nn.View(-1, 512)(img_Att2)	    						-- [batch_size, 512]
  local img_Att4 = nn.Linear(512, 512)(img_Att3):annotate{name='MLP3'}

  table.insert(outputs, img_Att4)
  return nn.gModule(inputs, outputs)
end

function attention.conv_att_best2()
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local hidden = inputs[2] --bs*512
  local conv5_ft = inputs[1]	-- [batch_size*14*14, 512] , conv5 pool output feature,

  local conv5_ft1 = nn.BatchNormalization(512, 1e-5, 0.1, true)(nn.Linear(512, 512)(nn.View(-1, 512)(conv5_ft)):annotate{name='MLP1'})
  local conv5_ft2 = nn.View(-1, 196, 512)(conv5_ft1) --[batch_size,14*14, 512]
  --512-dimensional feature mask vector
  local hidden_mask = nn.BatchNormalization(512, 1e-5, 0.1, true)(nn.Linear(512, 512)(hidden):annotate{name='MLP2'})
  local hidden_mask1 = nn.Sigmoid()(hidden_mask) -- mask_t
  local hidden_mask2 = nn.View(-1, 512, 1)(hidden_mask1) -- bs*512*1
  --probability distribution
  local p1 = nn.View(-1, 196)(nn.MM(false, false)({conv5_ft2, hidden_mask2})):annotate{name='MM1'} --bs*196*1
  --local p1_att = nn.SoftMax()(nn.Sigmoid()(p1)) --
  local p1_att = nn.SoftMax()(p1):annotate{name='SoftMax'} --
  local p1_att1 = nn.View(-1, 1, 196)(p1_att) --bs*1*196
  local img_Att2 = nn.MM(false, false)({p1_att1, conv5_ft2}):annotate{name='MM2'} --bs*1*512
  local img_Att3 = nn.View(-1, 512)(img_Att2)	    						-- [batch_size, 512]
  local img_Att4 = nn.Sigmoid()(nn.BatchNormalization(512, 1e-5, 0.1, true)(nn.Linear(512, 512)(img_Att3):annotate{name='MLP3'}))

  table.insert(outputs, img_Att4)
  return nn.gModule(inputs, outputs)
end

--[[
https://github.com/reedscot/cvpr2016/blob/master/modules/DocumentEncoder.lua
]]
function attention:CapEncoderEmb()
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- img
  local captions = inputs[1]--bs, 18, 512
  local x = nn.Transpose({2,3})(captions)--bs,512, 18
  -- 512 x alphasize
  local conv1 = nn.TemporalConvolution(18, 256, 7)(x)
  local thresh1 = nn.Threshold()(conv1)
  local max1 = nn.TemporalMaxPooling(3, 3)(thresh1)
  local conv2 = nn.TemporalConvolution(256, 256, 7)(max1)-- 168 x 256
  local thresh2 = nn.Threshold()(conv2)
  local max2 = nn.TemporalMaxPooling(3, 3)(thresh2) -- 54 x 256
  local conv3 = nn.TemporalConvolution(256, 256, 3)(max2)
  local thresh3 = nn.Threshold()(conv3)
  local max3 = nn.TemporalMaxPooling(3, 3)(thresh3)-- 17 x 256
  local conv4 = nn.TemporalConvolution(256, 256, 3)(max3)
  local thresh4 = nn.Threshold()(conv4)
  local max4 = nn.TemporalMaxPooling(3, 3)(thresh4)-- 5 x 256
  local conv5 = nn.TemporalConvolution(256, 256, 3)(max4)
  local thresh5 = nn.Threshold()(conv5)
  local rs = nn.Reshape(768)(thresh5)
  local fc2 = nn.Linear(768, 512)(rs)

  outputs = {}
  table.insert(outputs, fc2)
  return nn.gModule(inputs, outputs)
end


return attention
