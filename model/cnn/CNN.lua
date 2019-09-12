require 'cutorch'
require 'cudnn'
require 'cunn'
require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
require 'image'
local utils = require 'model.lm.utils'
local net_utils = require 'model.lm.net_utils'
local attention = require 'model.lm.attention'
local CNN = torch.class('CNN')
---------------------------------------------------------------------------
--
---------------------------------------------------------------------------
function CNN:__init(opt)
  self.start_cnn_from = utils.getopt(opt, 'start_cnn_from', 'model/cnn/cnn_conv_fc_defc_deconv.t7')
  self.start_cnn_fc_from = utils.getopt(opt, 'start_cnn_fc_from', 'model/cnn/cnn_conv_fc_defc_deconv.t7')
  self.cnn_vgg_proto = utils.getopt(opt, 'cnn_vgg_proto', 'model/cnn/VGG_ILSVRC_16_layers_deploy.prototxt')
  self.cnn_vgg_model = utils.getopt(opt, 'cnn_vgg_model', 'model/cnn/VGG_ILSVRC_16_layers.caffemodel')
  self.cnn_vgg_recons_proto = utils.getopt(opt, 'cnn_vgg_recons_proto', 'model/cnn/reconstruct/layer5_deploy.prototxt')
  self.cnn_vgg_recons_model = utils.getopt(opt, 'cnn_vgg_recons_model' ,'model/cnn/reconstruct/layer5.caffemodel')
end
---------------------------------------------------------------------------
--
---------------------------------------------------------------------------
function CNN:RecDeconv()
  local deconv = nn.Sequential()
  deconv:add(nn.SpatialFullConvolution(512, 512, 4, 4)) -- 512x4x4
  deconv:add(nn.SpatialBatchNormalization(512))
  deconv:add(nn.SpatialFullConvolution(512, 512, 4, 4, 2, 2)) -- 512x4x4->(4-1)*2-2*0+4+0=10
  deconv:add(nn.SpatialBatchNormalization(512))
  deconv:add(nn.SpatialFullConvolution(512, 512, 4, 4, 1, 1, 1, 1)) -- 512x10x10->(10-1)*1-2*0+4+1=11
  deconv:add(nn.SpatialBatchNormalization(512))
  deconv:add(nn.SpatialFullConvolution(512, 512, 4, 4, 1, 1, 1, 1)) -- 512x10x10->(10-1)*1-2*0+4+1=12
  deconv:add(nn.SpatialBatchNormalization(512))
  deconv:add(nn.SpatialFullConvolution(512, 512, 4, 4, 1, 1, 1, 1)) -- 512x10x10->(10-1)*1-2*0+4+1=1
  deconv:add(nn.SpatialBatchNormalization(512))
  deconv:add(nn.SpatialFullConvolution(512, 512, 4, 4, 1, 1, 1, 1)) -- 512x10x10->(10-1)*1-2*0+4+1=11
  deconv:add(nn.SpatialBatchNormalization(512))
  return deconv_rec
end
---------------------------------------------------------------------------
function CNN:CNN_SAVE()
  loaded_checkpoint = torch.load(self.start_cnn_from)
  lstmcnn = loaded_checkpoint.protos.cnn

  torch.save('model/cnn/vgg16_512_finetuned.t7', lstmcnn)
  collectgarbage()
end
---------------------------------------------------------------------------
function CNN:CNN_DECONV()
  local loss = {}

  if string.len(self.start_cnn_from) > 0 then
    local loaded_checkpoint = torch.load(self.start_cnn_from)
    local loadconv = loaded_checkpoint.conv
    local loadfc = loaded_checkpoint.fc
    local loaddefc = loaded_checkpoint.defc
    local loaddeconv = loaded_checkpoint.deconv

    conv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

    local conv = nn.Sequential()
    for i=1,4 do init_cnn_conv:add(loadconv:get(i)) end
    conv:add(init_conv1_2_pool_unpool)
    for i=6,9 do init_cnn_conv:add(loadconv:get(i)) end
    conv:add(init_conv2_2_pool_unpool)
    for i=11,16 do init_cnn_conv:add(loadconv:get(i)) end
    conv:add(init_conv3_3_pool_unpool)
    for i=18,23 do init_cnn_conv:add(loadconv:get(i)) end
    conv:add(init_conv4_3_pool_unpool)
    for i=25,30 do init_cnn_conv:add(loadconv:get(i)) end
    conv:add(init_conv5_3_pool_unpool)
    -----------------------------------------------------------------------------
    local fc = nn.Sequential()
    for i=1,9 do fc:add(loadfc:get(i)) end
    ----------------------------------------------------------------------------
    local defc = nn.Sequential()
    for i=1,8 do defc:add(loaddefc:get(i)) end
    ----------------------------------------------------------------------------
    deconv = nn.Sequential()
    deconv:add(nn.SpatialMaxUnpooling(conv5_3_pool_unpool))
    for i=2,6 do deconv:add(loaddeconv:get(i)) end
    deconv:add(nn.SpatialMaxUnpooling(conv4_3_pool_unpool))
    for i=8,14 do deconv:add(loaddeconv:get(i)) end
    deconv:add(nn.SpatialMaxUnpooling(conv3_3_pool_unpool))
    for i=16,21 do deconv:add(loaddeconv:get(i)) end
    deconv:add(nn.SpatialMaxUnpooling(conv2_2_pool_unpool))
    for i=23,26 do deconv:add(loaddeconv:get(i)) end
    deconv:add(nn.SpatialMaxUnpooling(conv1_2_pool_unpool))
    for i=28,30 do deconv:add(loaddeconv:get(i)) end
    collectgarbage()
    return conv, deconv, fc, defc

  elseif string.len(self.cnn_vgg_model) > 0 then
    local cnn = loadcaffe.load(self.cnn_vgg_recons_proto, self.cnn_vgg_recons_model, 'cudnn')

    conv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

    local conv = nn.Sequential()
    for i=1,4 do conv:add(nn.Sequential():add(cnn:get(i))) end
    conv:add(nn.Sequential():add(conv1_2_pool_unpool))
    for i=6,9 do conv:add(nn.Sequential():add(cnn:get(i))) end
    conv:add(nn.Sequential():add(conv2_2_pool_unpool))
    for i=11,16 do conv:add(nn.Sequential():add(cnn:get(i))) end
    conv:add(nn.Sequential():add(conv3_3_pool_unpool))
    for i=18,23 do conv:add(nn.Sequential():add(cnn:get(i))) end
    conv:add(nn.Sequential():add(conv4_3_pool_unpool))
    for i=25,30 do conv:add(nn.Sequential():add(cnn:get(i))) end
    conv:add(nn.Sequential():add(conv5_3_pool_unpool))

    ----------------------------------------------------------------------------
    fc = nn.Sequential()
    for i=57,63 do cnn_fc:add(nn.Sequential():add(cnn:get(i))) end
    fc:add(nn.Sequential():add(nn.Linear(4096, 512)))
    fc:add(nn.Sequential():add(cudnn.ReLU(true)))--relu

    ----------------------------------------------------------------------------
    defc = nn.Sequential()
    defc:add(nn.Sequential():add(nn.Linear(512, 4096)))
    defc:add(nn.Sequential():add(cudnn.ReLU(true)))
    defc:add(nn.Sequential():add(nn.Dropout(0.500000)))
    defc:add(nn.Sequential():add(nn.Linear(4096, 4096)))
    defc:add(nn.Sequential():add(cudnn.ReLU(true)))
    defc:add(nn.Sequential():add(nn.Dropout(0.500000)))
    defc:add(nn.Sequential():add(nn.Linear(4096, 25088)))
    defc:add(nn.Sequential():add(cudnn.ReLU(true)))
    --cnn_defc:add(nn.Sequential():add(nn.View(torch.LongStorage{-1, 512,7,7})))-- output 512x7x7
    ----------------------------------------------------------------------------

    deconv = nn.Sequential()
    deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv5_3_pool_unpool)))
    for i=32,36 do cnn_deconv:add(nn.Sequential():add(cnn:get(i))) end
    deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv4_3_pool_unpool)))
    for i=37,43 do cnn_deconv:add(nn.Sequential():add(cnn:get(i))) end
    deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv3_3_pool_unpool)))
    for i=44,49 do cnn_deconv:add(nn.Sequential():add(cnn:get(i))) end
    deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv2_2_pool_unpool)))
    for i=50,53 do cnn_deconv:add(nn.Sequential():add(cnn:get(i))) end
    deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv1_2_pool_unpool)))
    for i=54,56 do cnn_deconv:add(nn.Sequential():add(cnn:get(i))) end
    local cnn_model = {conv = conv, fc = fc, defc = defc, deconv = deconv}
    torch.save('model/cnn/cnn_conv_fc_defc_deconv.t7', cnn_model)
    collectgarbage()
    return conv, deconv, fc, defc
  end
end

function CNN:CNN()
  local loss = {}
  if string.len(self.start_cnn_from) > 0 then
    local loaded_checkpoint = torch.load(self.start_cnn_from)
    local loadconv = loaded_checkpoint.conv
    local loadfc = loaded_checkpoint.fc
    local loaddefc = loaded_checkpoint.defc
    local loaddeconv = loaded_checkpoint.deconv

    init_conv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()--pool1
    init_conv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    init_conv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    init_conv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    init_conv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

    init_cnn_conv = nn.Sequential()
    for i=1,4 do init_cnn_conv:add(loadconv:get(i)) end
    init_cnn_conv:add(init_conv1_2_pool_unpool)--conv1_2/pool
    for i=6,9 do init_cnn_conv:add(loadconv:get(i)) end
    init_cnn_conv:add(init_conv2_2_pool_unpool)--conv2_2/pool
    for i=11,16 do init_cnn_conv:add(loadconv:get(i)) end
    init_cnn_conv:add(init_conv3_3_pool_unpool) --conv3_3/pool
    for i=18,23 do init_cnn_conv:add(loadconv:get(i)) end
    init_cnn_conv:add(init_conv4_3_pool_unpool)--conv4_3/pool
    for i=25,30 do init_cnn_conv:add(loadconv:get(i)) end
    init_cnn_conv:add(init_conv5_3_pool_unpool)--conv5_3/pool

    -----------------------------------------------------------------------------
    init_cnn_fc = nn.Sequential()
    for i=1,9 do init_cnn_fc:add(loadfc:get(i)) end

    ----------------------------------------------------------------------------
    init_cnn_defc = nn.Sequential()
    for i=1,8 do init_cnn_defc:add(loaddefc:get(i)) end

    ----------------------------------------------------------------------------
    init_cnn_deconv = nn.Sequential()
    init_cnn_deconv:add(nn.SpatialMaxUnpooling(init_conv5_3_pool_unpool))
    for i=2,6 do init_cnn_deconv:add(loaddeconv:get(i)) end
    init_cnn_deconv:add(nn.SpatialMaxUnpooling(init_conv4_3_pool_unpool))
    for i=8,14 do init_cnn_deconv:add(loaddeconv:get(i)) end
    init_cnn_deconv:add(nn.SpatialMaxUnpooling(init_conv3_3_pool_unpool))
    for i=16,21 do init_cnn_deconv:add(loaddeconv:get(i)) end
    init_cnn_deconv:add(nn.SpatialMaxUnpooling(init_conv2_2_pool_unpool))
    for i=23,26 do init_cnn_deconv:add(loaddeconv:get(i)) end
    init_cnn_deconv:add(nn.SpatialMaxUnpooling(init_conv1_2_pool_unpool))
    for i=28,30 do init_cnn_deconv:add(loaddeconv:get(i)) end

    return init_cnn_conv, init_cnn_deconv, init_cnn_fc, init_cnn_defc

  elseif string.len(self.cnn_vgg_model) > 0 then
    local cnn = loadcaffe.load(self.cnn_vgg_recons_proto, self.cnn_vgg_recons_model, 'cudnn')

    conv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
    conv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

    cnn_conv = nn.Sequential()
    for i=1,4 do cnn_conv:add(cnn:get(i)) end
    cnn_conv:add(nn.Sequential():add(conv1_2_pool_unpool))
    for i=6,9 do cnn_conv:add(cnn:get(i)) end
    cnn_conv:add(nn.Sequential():add(conv2_2_pool_unpool))
    for i=11,16 do cnn_conv:add(cnn:get(i)) end
    cnn_conv:add(nn.Sequential():add(conv3_3_pool_unpool))
    for i=18,23 do cnn_conv:add(cnn:get(i)) end
    cnn_conv:add(nn.Sequential():add(conv4_3_pool_unpool))
    for i=25,30 do cnn_conv:add(cnn:get(i)) end
    cnn_conv:add(nn.Sequential():add(conv5_3_pool_unpool))

    ----------------------------------------------------------------------------
    cnn_fc = nn.Sequential()
    for i=57,63 do cnn_fc:add(cnn:get(i)) end
    cnn_fc:add(nn.Sequential():add(nn.Linear(4096, 512)))
    cnn_fc:add(nn.Sequential():add(cudnn.ReLU(true)))--relu

    ----------------------------------------------------------------------------
    cnn_defc = nn.Sequential()
    cnn_defc:add(nn.Sequential():add(nn.Linear(512, 4096)))
    cnn_defc:add(nn.Sequential():add(cudnn.ReLU(true)))			--fc7/relu; relu7
    cnn_defc:add(nn.Sequential():add(nn.Dropout(0.500000)))		--drop7
    cnn_defc:add(nn.Sequential():add(nn.Linear(4096, 4096)))	--fc7
    cnn_defc:add(nn.Sequential():add(cudnn.ReLU(true)))			--fc6/relu; relu6
    cnn_defc:add(nn.Sequential():add(nn.Dropout(0.500000)))		--drop6
    cnn_defc:add(nn.Sequential():add(nn.Linear(4096, 25088)))	--fc6,nn.Linear(25088, 4096)
    cnn_defc:add(nn.Sequential():add(cudnn.ReLU(true)))			--relu
    --cnn_defc:add(nn.Sequential():add(nn.View(torch.LongStorage{-1, 512,7,7})))-- output 512x7x7
    ----------------------------------------------------------------------------

    cnn_deconv = nn.Sequential()
    cnn_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv5_3_pool_unpool)))
    for i=32,36 do cnn_deconv:add(cnn:get(i)) end
    cnn_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv4_3_pool_unpool)))
    for i=37,43 do cnn_deconv:add(cnn:get(i)) end
    cnn_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv3_3_pool_unpool)))
    for i=44,49 do cnn_deconv:add(cnn:get(i)) end
    cnn_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv2_2_pool_unpool)))
    for i=50,53 do cnn_deconv:add(cnn:get(i)) end
    cnn_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv1_2_pool_unpool)))
    for i=54,56 do cnn_deconv:add(cnn:get(i)) end
    local cnn_model = { conv = cnn_conv,fc = cnn_fc,defc = cnn_defc,deconv = cnn_deconv}
    torch.save('model/cnn/cnn_conv_fc_defc_deconv.t7', cnn_model)
    collectgarbage()
    return cnn_conv, cnn_deconv, cnn_fc, cnn_defc
  end
end

-- load fc
function CNN:CNN_fc7()
  local cnn_vgg_recons_proto = 'model/cnn/reconstruct/layer7_deploy.prototxt'
  local cnn_vgg_recons_model = 'model/cnn/reconstruct/layer7.caffemodel'
  cnn = loadcaffe.load(cnn_vgg_recons_proto, cnn_vgg_recons_model, 'cudnn')

  conv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  local conv = nn.Sequential()
  for i=1,4 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  conv:add(nn.Sequential():add(conv1_2_pool_unpool))	--conv1_2/pool
  for i=6,9 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  conv:add(nn.Sequential():add(conv2_2_pool_unpool))	--conv2_2/pool
  for i=11,16 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  conv:add(nn.Sequential():add(conv3_3_pool_unpool))	--conv3_3/pool
  for i=18,23 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  conv:add(nn.Sequential():add(conv4_3_pool_unpool))	--conv4_3/pool
  for i=25,30 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  conv:add(nn.Sequential():add(conv5_3_pool_unpool))	--conv5_3/pool
  ----------------------------------------------------------------------------
  local fc = nn.Sequential()
  for i=32,36 do cnn_fc:add(cnn:get(i)) end
  ---------------------------------------------------------------------------
  cnn_defc = nn.Sequential()
  for i=37,40 do cnn_fc:add(cnn:get(i)) end
  --cnn_defc:add(nn.View(-1, 512,7,7))-- output 512x7x7
  --cnn_defc:add(nn.Sequential():add(nn.View(torch.LongStorage{-1, 512,7,7})))
  ----------------------------------------------------------------------------
  local deconv = nn.Sequential()
  deconv:add(nn.SpatialMaxUnpooling(conv5_3_pool_unpool))
  for i=41,45 do cnn_deconv:add(cnn:get(i)) end
  deconv:add(nn.SpatialMaxUnpooling(conv4_3_pool_unpool))
  for i=46,52 do cnn_deconv:add(cnn:get(i)) end
  deconv:add(nn.SpatialMaxUnpooling(conv3_3_pool_unpool))
  for i=53,58 do cnn_deconv:add(cnn:get(i)) end
  deconv:add(nn.SpatialMaxUnpooling(conv2_2_pool_unpool))
  for i=59,62 do cnn_deconv:add(cnn:get(i)) end
  deconv:add(nn.SpatialMaxUnpooling(conv1_2_pool_unpool))
  for i=63,65 do cnn_deconv:add(cnn:get(i)) end
  local cnn_model = {conv=conv,fc=fc,defc=defc,deconv=deconv}
  torch.save('model/cnn/cnn_conv_fc7_defc7_deconv.t7', cnn_model)

  return cnn_conv, cnn_deconv, cnn_fc, cnn_defc
end

-- load cnn model from pretrained model
function CNN:CNN_ReInit(cnn_checkpoint)

  local loadconv = cnn_checkpoint.conv
  local loadfc = cnn_checkpoint.fc
  local loaddefc = cnn_checkpoint.defc
  local loaddeconv = cnn_checkpoint.deconv

  conv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()--pool1
  conv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  local conv = nn.Sequential()
  for i=1,4 do conv:add(nn.Sequential():add(loadconv:get(i))) end
  conv:add(nn.Sequential():add(conv1_2_pool_unpool))--conv1_2/pool
  for i=6,9 do conv:add(nn.Sequential():add(loadconv:get(i))) end
  conv:add(nn.Sequential():add(conv2_2_pool_unpool))--conv2_2/pool
  for i=11,16 do conv:add(nn.Sequential():add(loadconv:get(i))) end
  conv:add(nn.Sequential():add(conv3_3_pool_unpool)) --conv3_3/pool
  for i=18,23 do conv:add(nn.Sequential():add(loadconv:get(i))) end
  conv:add(nn.Sequential():add(conv4_3_pool_unpool))--conv4_3/pool
  for i=25,30 do conv:add(nn.Sequential():add(loadconv:get(i))) end
  conv:add(nn.Sequential():add(conv5_3_pool_unpool))--conv5_3/pool
  -----------------------------------------------------------------------------
  fc = loadfc
  defc = loaddefc
  ----------------------------------------------------------------------------
  deconv = nn.Sequential()
  deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv5_3_pool_unpool)))
  for i=2,6 do deconv:add(nn.Sequential():add(loaddeconv:get(i))) end
  deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv4_3_pool_unpool)))
  for i=8,14 do deconv:add(nn.Sequential():add(loaddeconv:get(i))) end
  deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv3_3_pool_unpool)))
  for i=16,21 do deconv:add(nn.Sequential():add(loaddeconv:get(i))) end
  deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv2_2_pool_unpool)))
  for i=23,26 do deconv:add(nn.Sequential():add(loaddeconv:get(i))) end
  deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv1_2_pool_unpool)))
  for i=28,30 do deconv:add(nn.Sequential():add(loaddeconv:get(i))) end

  return conv, deconv, fc, defc
end

function CNN:LoadtPretrainedCNN(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local loadconv = loaded_checkpoint.conv
  local loadfc = loaded_checkpoint.fc
  local loaddefc = loaded_checkpoint.defc
  local loaddeconv = loaded_checkpoint.deconv

  conv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()--pool1
  conv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  local conv = nn.Sequential()
  for i=1,4 do conv:add(loadconv:get(i)) end
  conv:add(init_conv1_2_pool_unpool)
  for i=6,9 do conv:add(loadconv:get(i)) end
  conv:add(init_conv2_2_pool_unpool)
  for i=11,16 do conv:add(loadconv:get(i)) end
  conv:add(init_conv3_3_pool_unpool)
  for i=18,23 do conv:add(loadconv:get(i)) end
  conv:add(init_conv4_3_pool_unpool)
  for i=25,30 do conv:add(loadconv:get(i)) end
  conv:add(init_conv5_3_pool_unpool)
  -----------------------------------------------------------------------------
  fc = nn.Sequential()
  for i=1,9 do fc:add(loadfc:get(i)) end
  ----------------------------------------------------------------------------
  defc = nn.Sequential()
  for i=1,8 do defc:add(loaddefc:get(i)) end
  ----------------------------------------------------------------------------
  deconv = nn.Sequential()
  deconv:add(nn.SpatialMaxUnpooling(init_conv5_3_pool_unpool))
  for i=2,6 do deconv:add(loaddeconv:get(i)) end
  deconv:add(nn.SpatialMaxUnpooling(init_conv4_3_pool_unpool))
  for i=8,14 do deconv:add(loaddeconv:get(i)) end
  deconv:add(nn.SpatialMaxUnpooling(init_conv3_3_pool_unpool))
  for i=16,21 do deconv:add(loaddeconv:get(i)) end
  deconv:add(nn.SpatialMaxUnpooling(init_conv2_2_pool_unpool))
  for i=23,26 do deconv:add(loaddeconv:get(i)) end
  deconv:add(nn.SpatialMaxUnpooling(init_conv1_2_pool_unpool))
  for i=28,30 do deconv:add(loaddeconv:get(i)) end
  collectgarbage()
  return conv, deconv, fc, defc
end


function CNN:LoadCNN_CONV_DECONV_CONV5_POOLOUT(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  conv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()--pool1
  conv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv5_3_pool_unpool)	--conv5_3/pool
  lstm_conv:add(lstmcnn:get(32))

  ----------------------------------------------------------------------------
  local cnn_checkpoint = torch.load('model/cnn/cnn_conv_fc7_defc7_deconv.t7')
  local lstmdefc = cnn_checkpoint.defc
  local lstmdeconv = cnn_checkpoint.deconv

  lstm_deconv = nn.Sequential()
  --lstm_deconv:add(cudnn.ReLU(true))
  lstm_deconv:add(nn.Linear(512, 4096))
  lstm_deconv:add(cudnn.ReLU(true))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmdefc:get(1).weight
  layer.bias = lstmdefc:get(1).bias
  lstm_deconv:add(nn.Sequential():add(layer))
  lstm_deconv:add(nn.Sequential():add(cudnn.ReLU(true)))
  layer = nn.Linear(4096, 25088)
  layer.weight = lstmdefc:get(3).weight
  layer.bias = lstmdefc:get(3).bias
  lstm_deconv:add(nn.Sequential():add(layer))
  lstm_deconv:add(nn.Sequential():add(cudnn.ReLU(true)))

  return lstm_conv, lstm_deconv
end


function CNN:LoadCNN512(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv5_3_pool_unpool)	--conv5_3/pool
  lstm_conv:add(lstmcnn:get(32))
  layer = nn.Linear(25088, 4096)
  layer.weight = lstmcnn:get(33).weight
  layer.bias = lstmcnn:get(33).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_conv:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmcnn:get(36).weight
  layer.bias = lstmcnn:get(36).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_conv:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 512)
  layer.weight = lstmcnn:get(39).weight
  layer.bias = lstmcnn:get(39).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))

  return lstm_conv
end

function CNN:LoadGoogleV3(start_cnn_from)
  local googlenet_v3 = torch.load(start_cnn_from)

  local fc = nn.Sequential()
  fc:add(nn.Linear(1008,512))
  fc:add(cudnn.ReLU(true))
  return googlenet_v3, fc
end

function CNN:LoadResNet152(start_cnn_from)
  local model = torch.load(start_cnn_from)
  model:remove(#model.modules) --2048

  local fc = nn.Sequential()
  fc:add(nn.Linear(2048,512))
  fc:add(cudnn.ReLU(true))
  return model, fc
end


function CNN:LoadCNN_CONV5_CAFFE(cnn_proto, cnn_model)
  local cnn = loadcaffe.load(cnn_proto, cnn_model, 'cudnn')

  conv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  conv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  cnn_conv = nn.Sequential()
  for i=1,4 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  cnn_conv:add(nn.Sequential():add(conv1_2_pool_unpool))	--conv1_2/pool
  for i=6,9 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  cnn_conv:add(nn.Sequential():add(conv2_2_pool_unpool))	--conv2_2/pool
  for i=11,16 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  cnn_conv:add(nn.Sequential():add(conv3_3_pool_unpool))	--conv3_3/pool
  for i=18,23 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  cnn_conv:add(nn.Sequential():add(conv4_3_pool_unpool))	--conv4_3/pool
  for i=25,30 do cnn_conv:add(nn.Sequential():add(cnn:get(i))) end
  --cnn_conv:add(nn.Sequential():add(conv5_3_pool_unpool))	--conv5_3/pool
  cnn_conv:add(nn.Sequential():add(cnn:get(32)))--torch_view
end


function CNN:LoadCNN_CONV5(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmcnn:get(32))

  return lstm_conv
end


function CNN:LoadCNN_CONV(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmcnn:get(32))

  lstm_fc = nn.Sequential()
  lstm_fc:add(nn.Sequential():add(nn.View(torch.LongStorage{-1, 512,14,14})))-- output 512x7x7
  lstm_fc:add(lstmconv5_3_pool_unpool)	--conv5_3/pool
  lstm_fc:add(nn.Sequential():add(nn.View(-1):setNumInputDims(3)))
  layer = nn.Linear(25088, 4096)
  layer.weight = lstmcnn:get(33).weight
  layer.bias = lstmcnn:get(33).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmcnn:get(36).weight
  layer.bias = lstmcnn:get(36).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 512)
  layer.weight = lstmcnn:get(39).weight
  layer.bias = lstmcnn:get(39).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))

  return lstm_conv, lstm_fc
end


function CNN:LoadCNN_CONV_FC512(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv5_3_pool_unpool)	--conv5_3/pool
  lstm_conv:add(lstmcnn:get(32))
  layer = nn.Linear(25088, 4096)
  layer.weight = lstmcnn:get(33).weight
  layer.bias = lstmcnn:get(33).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_conv:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmcnn:get(36).weight
  layer.bias = lstmcnn:get(36).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_conv:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 512)
  layer.weight = lstmcnn:get(39).weight
  layer.bias = lstmcnn:get(39).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))

  return lstm_conv
end


function CNN:LoadCNN_CONV_FC512_DECONV(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv5_3_pool_unpool)	--conv5_3/pool
  lstm_conv:add(lstmcnn:get(32))
  layer = nn.Linear(25088, 4096)
  layer.weight = lstmcnn:get(33).weight
  layer.bias = lstmcnn:get(33).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_conv:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmcnn:get(36).weight
  layer.bias = lstmcnn:get(36).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_conv:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 512)
  layer.weight = lstmcnn:get(39).weight
  layer.bias = lstmcnn:get(39).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))

  local cnn_checkpoint = torch.load('model/cnn/cnn_conv_fc7_defc7_deconv.t7')
  local lstmdefc = cnn_checkpoint.defc
  local lstmdeconv = cnn_checkpoint.deconv

  lstm_deconv = nn.Sequential()
  lstm_deconv:add(nn.Linear(512, 4096))
  lstm_deconv:add(cudnn.ReLU(true))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmdefc:get(1).weight
  layer.bias = lstmdefc:get(1).bias
  lstm_deconv:add(nn.Sequential():add(layer))
  lstm_deconv:add(nn.Sequential():add(cudnn.ReLU(true)))
  layer = nn.Linear(4096, 25088)
  layer.weight = lstmdefc:get(3).weight
  layer.bias = lstmdefc:get(3).bias
  lstm_deconv:add(nn.Sequential():add(layer))
  lstm_deconv:add(nn.Sequential():add(cudnn.ReLU(true)))

  return lstm_conv, lstm_deconv
end


--[[
I use this for lm cnn train
]]
function CNN:LoadCNN_CONV_FC1024(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmcnn:get(32))

  lstm_fc = nn.Sequential()
  lstm_fc:add(nn.Sequential():add(nn.View(torch.LongStorage{-1, 512,14,14})))-- output 512x7x7
  lstm_fc:add(lstmconv5_3_pool_unpool)	--conv5_3/pool
  lstm_fc:add(nn.Sequential():add(nn.View(-1):setNumInputDims(3)))
  layer = nn.Linear(25088, 4096)
  layer.weight = lstmcnn:get(33).weight
  layer.bias = lstmcnn:get(33).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmcnn:get(36).weight
  layer.bias = lstmcnn:get(36).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 1024)
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))

  return lstm_conv, lstm_fc
end

function CNN:CNN_test(conv, deconv, image_path)

  local img = image.load(image_path,3,'byte')
  test = torch.CudaTensor(1,3,img:size(2),img:size(3)):fill(1)
  input = torch.CudaTensor(1,3,img:size(2),img:size(3))
  input[{1}]=img
  defc7 = attention.defc7()
  defc7:cuda()
  --input= net_utils.prepro(input, false, 0)
  convft = conv:forward(input)
  local outdefc7 = defc7:forward(convft)
  deconvft = deconv:forward(outdefc7)

  local sum_images = image.toDisplayTensor(deconvft[1],1,1)
  image.save('tmp_out0.png', sum_images)

  local vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  vgg_mean = vgg_mean:typeAs(input)
  deconvft:add(1, vgg_mean:expandAs(deconvft))
  image.save('tmp_out.png', deconvft[1])
  image.save('tmp_in.png', img)

  print('Test finish')
end

function CNN:LoadCNN_DECONV5(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv5_3_pool_unpool)	--conv5_3/pool

  lstm_fc = nn.Sequential()
  lstm_fc:add(nn.Sequential():add(nn.View(-1):setNumInputDims(3)))
  layer = nn.Linear(25088, 4096)
  layer.weight = lstmcnn:get(33).weight
  layer.bias = lstmcnn:get(33).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmcnn:get(36).weight
  layer.bias = lstmcnn:get(36).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 512)
  layer.weight = lstmcnn:get(39).weight
  layer.bias = lstmcnn:get(39).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))

  ----------------------------------------------------------------------------
  lstm_defc = nn.Sequential()
  lstm_defc:add(nn.Sequential():add(nn.Linear(512, 4096)))
  lstm_defc:add(nn.Sequential():add(cudnn.ReLU(true)))			--fc7/relu; relu7
  lstm_defc:add(nn.Sequential():add(nn.Dropout(0.500000)))		--drop7
  lstm_defc:add(nn.Sequential():add(nn.Linear(4096, 4096)))	--fc7
  lstm_defc:add(nn.Sequential():add(cudnn.ReLU(true)))			--fc6/relu; relu6
  lstm_defc:add(nn.Sequential():add(nn.Dropout(0.500000)))		--drop6
  lstm_defc:add(nn.Sequential():add(nn.Linear(4096, 25088)))	--fc6,nn.Linear(25088, 4096)
  lstm_defc:add(nn.Sequential():add(cudnn.ReLU(true)))			--relu
  --cnn_defc:add(nn.Sequential():add(nn.View(torch.LongStorage{-1, 512,7,7})))-- output 512x7x7


  local cnn_vgg_recons_proto = 'model/cnn/reconstruct/layer7_deploy.prototxt'
  local cnn_vgg_recons_model = 'model/cnn/reconstruct/layer7.caffemodel'
  cnn = loadcaffe.load(cnn_vgg_recons_proto, cnn_vgg_recons_model, 'cudnn')
  lstm_deconv = nn.Sequential()
  lstm_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv5_3_pool_unpool)))--dec:conv5_3/unpool
  for i=41,45 do lstm_deconv:add(nn.Sequential():add(cnn:get(i))) end
  cnn_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv4_3_pool_unpool)))--dec:conv4_3/unpool
  for i=46,52 do lstm_deconv:add(nn.Sequential():add(cnn:get(i))) end
  lstm_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv3_3_pool_unpool)))--dec:conv3_3/unpool
  for i=53,58 do lstm_deconv:add(nn.Sequential():add(cnn:get(i))) end
  lstm_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv2_2_pool_unpool)))--dec:conv2_2/unpool
  for i=59,62 do lstm_deconv:add(nn.Sequential():add(cnn:get(i))) end
  lstm_deconv:add(nn.Sequential():add(nn.SpatialMaxUnpooling(conv1_2_pool_unpool)))--dec:conv1_2/unpool(27)
  for i=63,65 do lstm_deconv:add(nn.Sequential():add(cnn:get(i))) end

  return lstm_conv, lstm_deconv, lstm_fc, lstm_defc
end



function CNN:LoadCNN_DEFC7_INIT(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local init_conv = loaded_checkpoint.protos.cnn.conv
  local init_fc = loaded_checkpoint.protos.cnn.fc
  local init_defc = loaded_checkpoint.protos.cnn.defc
  local init_deconv = loaded_checkpoint.protos.cnn.deconv

  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(1).weight
  layer.bias = init_conv:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(3).weight
  layer.bias = init_conv:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(6).weight
  layer.bias = init_conv:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(8).weight
  layer.bias = init_conv:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(11).weight
  layer.bias = init_conv:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(13).weight
  layer.bias = init_conv:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(15).weight
  layer.bias = init_conv:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(18).weight
  layer.bias = init_conv:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(20).weight
  layer.bias = init_conv:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(22).weight
  layer.bias = init_conv:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(25).weight
  layer.bias = init_conv:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(27).weight
  layer.bias = init_conv:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = init_conv:get(29).weight
  layer.bias = init_conv:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv5_3_pool_unpool)	--conv5_3/pool

  lstm_fc = nn.Sequential()
  for i=1,9 do lstm_fc:add(init_fc:get(i)) end

  ----------------------------------------------------------------------------
  lstm_defc = nn.Sequential()
  layer = nn.Linear(4096, 4096)
  layer.weight = init_defc:get(1).weight
  layer.bias = init_defc:get(1).bias
  lstm_defc:add(nn.Sequential():add(layer))
  lstm_defc:add(nn.Sequential():add(cudnn.ReLU(true)))
  layer = nn.Linear(4096, 25088)
  layer.weight = init_defc:get(3).weight
  layer.bias = init_defc:get(3).bias
  lstm_defc:add(nn.Sequential():add(layer))
  lstm_defc:add(nn.Sequential():add(cudnn.ReLU(true)))

  lstm_deconv = nn.Sequential()
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv5_3_pool_unpool))--dec:conv5_3/unpool
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(2).weight
  layer.bias = init_deconv:get(2).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(4).weight
  layer.bias = init_deconv:get(4).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(6).weight
  layer.bias = init_deconv:get(6).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv4_3_pool_unpool))--dec:conv4_3/unpool
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(9).weight
  layer.bias = init_deconv:get(9).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(11).weight
  layer.bias = init_deconv:get(11).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(13).weight
  layer.bias = init_deconv:get(13).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv3_3_pool_unpool))--dec:conv4_3/unpool
  layer = cudnn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(16).weight
  layer.bias = init_deconv:get(16).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(18).weight
  layer.bias = init_deconv:get(18).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(256, 128, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(20).weight
  layer.bias = init_deconv:get(20).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv2_2_pool_unpool))--dec:conv4_3/unpool
  layer = nn.SpatialFullConvolution(128, 128, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(23).weight
  layer.bias = init_deconv:get(23).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(128, 64, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(25).weight
  layer.bias = init_deconv:get(25).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv1_2_pool_unpool))--dec:conv4_3/unpool
  layer = nn.SpatialFullConvolution(64, 64, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(28).weight
  layer.bias = init_deconv:get(28).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(64, 3, 3, 3, 1, 1, 1, 1)
  layer.weight = init_deconv:get(30).weight
  layer.bias = init_deconv:get(30).bias
  lstm_deconv:add(layer)

  return lstm_conv, lstm_deconv, lstm_fc, lstm_defc
end

function CNN:LoadCNN_DEFC7(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv5_3_pool_unpool)	--conv5_3/pool

  lstm_fc = nn.Sequential()
  lstm_fc:add(nn.Sequential():add(nn.View(-1):setNumInputDims(3)))
  layer = nn.Linear(25088, 4096)
  layer.weight = lstmcnn:get(33).weight
  layer.bias = lstmcnn:get(33).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmcnn:get(36).weight
  layer.bias = lstmcnn:get(36).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 512)
  layer.weight = lstmcnn:get(39).weight
  layer.bias = lstmcnn:get(39).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))

  ----------------------------------------------------------------------------
  local cnn_checkpoint = torch.load('model/cnn/cnn_conv_fc7_defc7_deconv.t7')
  local lstmdefc = cnn_checkpoint.defc

  lstm_defc = nn.Sequential()
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmdefc:get(1).weight
  layer.bias = lstmdefc:get(1).bias
  lstm_defc:add(nn.Sequential():add(layer))
  lstm_defc:add(nn.Sequential():add(cudnn.ReLU(true)))
  layer = nn.Linear(4096, 25088)
  layer.weight = lstmdefc:get(3).weight
  layer.bias = lstmdefc:get(3).bias
  lstm_defc:add(nn.Sequential():add(layer))
  lstm_defc:add(nn.Sequential():add(cudnn.ReLU(true)))

  local lstmdeconv = cnn_checkpoint.deconv

  lstm_deconv = nn.Sequential()
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv5_3_pool_unpool))--dec:conv5_3/unpool
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(2).weight
  layer.bias = lstmdeconv:get(2).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(4).weight
  layer.bias = lstmdeconv:get(4).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(6).weight
  layer.bias = lstmdeconv:get(6).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv4_3_pool_unpool))--dec:conv4_3/unpool
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(9).weight
  layer.bias = lstmdeconv:get(9).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(11).weight
  layer.bias = lstmdeconv:get(11).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(13).weight
  layer.bias = lstmdeconv:get(13).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv3_3_pool_unpool))--dec:conv4_3/unpool
  layer = cudnn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(16).weight
  layer.bias = lstmdeconv:get(16).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(18).weight
  layer.bias = lstmdeconv:get(18).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(256, 128, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(20).weight
  layer.bias = lstmdeconv:get(20).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv2_2_pool_unpool))--dec:conv4_3/unpool
  layer = nn.SpatialFullConvolution(128, 128, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(23).weight
  layer.bias = lstmdeconv:get(23).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(128, 64, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(25).weight
  layer.bias = lstmdeconv:get(25).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv1_2_pool_unpool))--dec:conv4_3/unpool
  layer = nn.SpatialFullConvolution(64, 64, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(28).weight
  layer.bias = lstmdeconv:get(28).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(64, 3, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(30).weight
  layer.bias = lstmdeconv:get(30).bias
  lstm_deconv:add(layer)

  return lstm_conv, lstm_deconv, lstm_fc, lstm_defc
end


function CNN:LoadCNN(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv5_3_pool_unpool)	--conv5_3/pool
  lstm_conv:add(lstmcnn:get(32))
  layer = nn.Linear(25088, 4096)
  layer.weight = lstmcnn:get(33).weight
  layer.bias = lstmcnn:get(33).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_conv:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmcnn:get(36).weight
  layer.bias = lstmcnn:get(36).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_conv:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 512)
  layer.weight = lstmcnn:get(39).weight
  layer.bias = lstmcnn:get(39).bias
  lstm_conv:add(nn.Sequential():add(layer))
  lstm_conv:add(nn.Sequential():add(cudnn.ReLU(true)))

  ----------------------------------------------------------------------------
  local cnn_checkpoint = torch.load('model/cnn/cnn_conv_fc7_defc7_deconv.t7')
  local lstmdefc = cnn_checkpoint.defc
  local lstmdeconv = cnn_checkpoint.deconv

  lstm_deconv = nn.Sequential()
  lstm_deconv:add(cudnn.ReLU(true))
  lstm_deconv:add(nn.Linear(512, 4096))
  lstm_deconv:add(cudnn.ReLU(true))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmdefc:get(1).weight
  layer.bias = lstmdefc:get(1).bias
  lstm_deconv:add(nn.Sequential():add(layer))
  lstm_deconv:add(nn.Sequential():add(cudnn.ReLU(true)))
  layer = nn.Linear(4096, 25088)
  layer.weight = lstmdefc:get(3).weight
  layer.bias = lstmdefc:get(3).bias
  lstm_deconv:add(nn.Sequential():add(layer))
  lstm_deconv:add(nn.Sequential():add(cudnn.ReLU(true)))
  --lstm_deconv:add(nn.Sequential():add(nn.Reshape(-1, 512,7,7)))-- output 512x7x7
  lstm_deconv:add(nn.Sequential():add(nn.View(torch.LongStorage{-1, 512,7,7})))-- output 512x7x7
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv5_3_pool_unpool))--dec:conv5_3/unpool
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(2).weight
  layer.bias = lstmdeconv:get(2).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(4).weight
  layer.bias = lstmdeconv:get(4).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(6).weight
  layer.bias = lstmdeconv:get(6).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv4_3_pool_unpool))--dec:conv4_3/unpool
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(9).weight
  layer.bias = lstmdeconv:get(9).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(11).weight
  layer.bias = lstmdeconv:get(11).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(13).weight
  layer.bias = lstmdeconv:get(13).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv3_3_pool_unpool))--dec:conv4_3/unpool
  layer = cudnn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(16).weight
  layer.bias = lstmdeconv:get(16).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(18).weight
  layer.bias = lstmdeconv:get(18).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(256, 128, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(20).weight
  layer.bias = lstmdeconv:get(20).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv2_2_pool_unpool))--dec:conv4_3/unpool
  layer = nn.SpatialFullConvolution(128, 128, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(23).weight
  layer.bias = lstmdeconv:get(23).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(128, 64, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(25).weight
  layer.bias = lstmdeconv:get(25).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv1_2_pool_unpool))--dec:conv4_3/unpool
  layer = nn.SpatialFullConvolution(64, 64, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(28).weight
  layer.bias = lstmdeconv:get(28).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(64, 3, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(30).weight
  layer.bias = lstmdeconv:get(30).bias
  lstm_deconv:add(layer)

  return lstm_conv, lstm_deconv
end

--[[
used for attention and deconvolution
]]
function CNN:LoadCNN_DEFC7_ATT(start_cnn_from)
  local loaded_checkpoint = torch.load(start_cnn_from)
  local lstmcnn = loaded_checkpoint.protos.cnn
  local layer

  lstmconv1_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv2_2_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv3_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv4_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()
  lstmconv5_3_pool_unpool = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()

  lstm_conv = nn.Sequential()
  layer = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(1).weight
  layer.bias = lstmcnn:get(1).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(3).weight
  layer.bias = lstmcnn:get(3).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv1_2_pool_unpool)	--conv1_2/pool
  layer = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(6).weight
  layer.bias = lstmcnn:get(6).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(8).weight
  layer.bias = lstmcnn:get(8).bias
  lstm_conv:add(layer)
  lstm_conv:add(cudnn.ReLU(true)):add(lstmconv2_2_pool_unpool)	--conv2_2/pool
  layer = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(11).weight
  layer.bias = lstmcnn:get(11).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(13).weight
  layer.bias = lstmcnn:get(13).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(15).weight
  layer.bias = lstmcnn:get(15).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv3_3_pool_unpool)	--conv3_3/pool
  layer = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(18).weight
  layer.bias = lstmcnn:get(18).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(20).weight
  layer.bias = lstmcnn:get(20).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(22).weight
  layer.bias = lstmcnn:get(22).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  lstm_conv:add(lstmconv4_3_pool_unpool)	--conv4_3/pool
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(25).weight
  layer.bias = lstmcnn:get(25).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(27).weight
  layer.bias = lstmcnn:get(27).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
  layer.weight = lstmcnn:get(29).weight
  layer.bias = lstmcnn:get(29).bias
  lstm_conv:add(layer):add(cudnn.ReLU(true))
  --lstm_conv:add(lstmconv5_3_pool_unpool)	--conv5_3/pool
  lstm_conv:add(lstmcnn:get(32))--bs,100352

  ----------------------------------------------------------------------------
  lstm_fc = nn.Sequential()
  lstm_fc:add(nn.Sequential():add(nn.View(torch.LongStorage{-1, 512,14,14})))-- output 512x14x14
  lstm_fc:add(lstmconv5_3_pool_unpool)	--conv5_3/pool
  lstm_fc:add(nn.Sequential():add(nn.View(-1):setNumInputDims(3)))
  layer = nn.Linear(25088, 4096)
  layer.weight = lstmcnn:get(33).weight
  layer.bias = lstmcnn:get(33).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmcnn:get(36).weight
  layer.bias = lstmcnn:get(36).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))
  lstm_fc:add(nn.Sequential():add(nn.Dropout(0.500000)))
  layer = nn.Linear(4096, 512)
  layer.weight = lstmcnn:get(39).weight
  layer.bias = lstmcnn:get(39).bias
  lstm_fc:add(nn.Sequential():add(layer))
  lstm_fc:add(nn.Sequential():add(cudnn.ReLU(true)))

  ----------------------------------------------------------------------------
  --[[
	local lstmdefc7 = nn.Sequential()
	lstmdefc7:add(nn.Linear(512, 512))
	lstmdefc7:add(cudnn.ReLU(true))
	lstmdefc7:add(nn.Dropout(0.500000))
	lstmdefc7:add(nn.Linear(512, 4096))
	]]
  ----------------------------------------------------------------------------
  local cnn_checkpoint = torch.load('model/cnn/cnn_conv_fc7_defc7_deconv.t7')
  local lstmdefc = cnn_checkpoint.defc

  lstm_defc = nn.Sequential()
  lstm_defc:add(nn.Linear(512, 4096))
  lstm_defc:add(cudnn.ReLU(true))
  layer = nn.Linear(4096, 4096)
  layer.weight = lstmdefc:get(1).weight
  layer.bias = lstmdefc:get(1).bias
  lstm_defc:add(nn.Sequential():add(layer))
  lstm_defc:add(nn.Sequential():add(cudnn.ReLU(true)))
  layer = nn.Linear(4096, 25088)
  layer.weight = lstmdefc:get(3).weight
  layer.bias = lstmdefc:get(3).bias
  lstm_defc:add(nn.Sequential():add(layer))
  lstm_defc:add(nn.Sequential():add(nn.View(-1, 512,7,7)))-- output 512x7x7
  lstm_defc:add(nn.Sequential():add(cudnn.ReLU(true)))
  --lstm_defc:add(nn.Sequential():add(nn.SpatialMaxUnpooling(lstmconv5_3_pool_unpool)))--dec:conv5_3/unpool --output 512,14,14
  lstm_defc:add(nn.View(-1):setNumInputDims(3)) --bs,100352

  local lstmdeconv = cnn_checkpoint.deconv

  lstm_deconv = nn.Sequential()
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(2).weight
  layer.bias = lstmdeconv:get(2).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(4).weight
  layer.bias = lstmdeconv:get(4).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(6).weight
  layer.bias = lstmdeconv:get(6).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv4_3_pool_unpool))--dec:conv4_3/unpool
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(9).weight
  layer.bias = lstmdeconv:get(9).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(11).weight
  layer.bias = lstmdeconv:get(11).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = cudnn.SpatialFullConvolution(512, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(13).weight
  layer.bias = lstmdeconv:get(13).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv3_3_pool_unpool))--dec:conv4_3/unpool
  layer = cudnn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(16).weight
  layer.bias = lstmdeconv:get(16).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(18).weight
  layer.bias = lstmdeconv:get(18).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(256, 128, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(20).weight
  layer.bias = lstmdeconv:get(20).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv2_2_pool_unpool))--dec:conv4_3/unpool
  layer = nn.SpatialFullConvolution(128, 128, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(23).weight
  layer.bias = lstmdeconv:get(23).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(128, 64, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(25).weight
  layer.bias = lstmdeconv:get(25).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  lstm_deconv:add(nn.SpatialMaxUnpooling(lstmconv1_2_pool_unpool))--dec:conv4_3/unpool
  layer = nn.SpatialFullConvolution(64, 64, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(28).weight
  layer.bias = lstmdeconv:get(28).bias
  lstm_deconv:add(layer):add(cudnn.ReLU(true))
  layer = nn.SpatialFullConvolution(64, 3, 3, 3, 1, 1, 1, 1)
  layer.weight = lstmdeconv:get(30).weight
  layer.bias = lstmdeconv:get(30).bias
  lstm_deconv:add(layer)

  return lstm_conv, lstm_deconv, lstm_fc, lstm_defc
end

--[[
cnn = CNN()
CNN.CNN_SAVE()
]]
