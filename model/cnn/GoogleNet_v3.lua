
require 'torch'
require 'image'
require 'nn'
local pl = require('pl.import_into')()

-- Some definitions copied from the TensorFlow model
-- input subtraction
local input_sub = 128
-- Scale input image
local input_scale = 0.0078125
-- input dimension
local input_dim = 299

local load_image = function(path)
  local img   = image.load(path, 3)
  local w, h  = img:size(3), img:size(2)
  local min   = math.min(w, h)
  img         = image.crop(img, 'c', min, min)
  img         = image.scale(img, input_dim)
  -- normalize image
  img:mul(255):add(-input_sub):mul(input_scale)
  -- due to batch normalization we must use minibatches
  return img:float():view(1, img:size(1), img:size(2), img:size(3))
end

require "cunn"
require "cudnn"

local net = torch.load('/home/ajax/gitnas/ntu_lib_utils/lua_utils/library/cls_caption/cls_cvpr2017/model/cnn/inceptionv3.net')

googlenet = nn.Sequential()		
for i=1,31 do googlenet:add(net:get(i)) end
torch.save('model/cnn/googlenet_v3.t7', googlenet)

net:evaluate()

local synsets = pl.utils.readlines('/home/ajax/gitnas/ntu_lib_utils/lua_utils/library/cls_caption/cls_cvpr2017/model/cnn/synsets.txt')

local img = load_image('/home/ajax/gitnas/ntu_lib_utils/lua_utils/library/cls_caption/cls_cvpr2017/model/cnn/googlenet/inception-v3.torch/cat.jpg')
img = img:cuda()
-- predict
local scores = net:forward(img)
scores = scores:float():squeeze()

-- find top5 matches
local _,ind = torch.sort(scores, true)
print('\nRESULTS (top-5):')
print('----------------')
for i=1,5 do
  local synidx = ind[i]
  print(string.format(
    "score = %f: %s (%d)", scores[ind[i]], synsets[synidx], ind[i]
  ))
end