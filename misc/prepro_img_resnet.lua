require 'nn'
require 'optim'
require 'torch'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
cjson=require('cjson') 
require 'xlua'
require 'cudnn'
local t = require 'transforms'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json','./data/data_prepro.json','path to the json file containing vocab and answers')
cmd:option('-image_root','./data/Images/mscoco/','path to the image root')

cmd:option('-residule_path', '/home/ajax/gitnas/ntu_lib_utils/lua_utils/library/cls_cnn/resnet/fb.resnet.torch/models/offcial_pretrained/resnet-101.t7')
cmd:option('-batch_size', 60, 'batch_size')

cmd:option('-out_name', 'data_resnet_101_img_ft.h5', 'output name')

cmd:option('-gpuid', 2, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)
local model = torch.load(opt.residule_path)
-- Remove the fully connected layer
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)

print(model)
model:evaluate()
model=model:cuda()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}


imloader={}
function imloader:load(fname)
    self.im=image.load(fname)
end
function loadim(imname)

    local img = image.load(imname, 3, 'float')
    img = transform(img)
    return img
end

local image_root = opt.image_root
-- open the mdf5 file

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(train_list, image_root .. imname)
end

local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
    table.insert(test_list, image_root .. imname)
end

local ndims=2048
local batch_size = opt.batch_size
local sz=#train_list
local feat_train=torch.FloatTensor(sz, 1, 2048) --ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.FloatTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(train_list[i+j-1])
    end
    -- Get the output of the layer before the (removed) fully connected layer
    local output = model:forward(ims:cuda()):squeeze(1)

    -- this is necesary because the model outputs different dimension based on size of input
    if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end 
    
    feat_train[{{i,r},{}}]=output:float()
    collectgarbage()
end

local ndims=2048
local batch_size = opt.batch_size
local sz=#test_list
local feat_test=torch.FloatTensor(sz,1, 2048)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.FloatTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(test_list[i+j-1])
    end
        -- Get the output of the layer before the (removed) fully connected layer
    local output = model:forward(ims:cuda()):squeeze(1)

    -- this is necesary because the model outputs different dimension based on size of input
    if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end 
    feat_test[{{i,r},{}}]=output:float()
    collectgarbage()
end

local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:close()   
