require 'cutorch'
require 'cudnn'
require 'cunn'
require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'image'
local LanguageEmbedding = require 'model.lm.LanguageEmbedding'
local DocumentEncoder = require 'model.nlp.modules.DocumentEncoderEmb'
--[[
local bs=16
local seq_len = 18
local emb_dim = 512
x=torch.Tensor(1,512,18)
local conv1 = nn.TemporalConvolution(18, 256, 7):forward(x)
local thresh1 = nn.Threshold():forward(conv1)
local max1 = nn.TemporalMaxPooling(3, 3):forward(thresh1)
-- 336 x 256
local conv2 = nn.TemporalConvolution(256, 256, 7):forward(max1)
local thresh2 = nn.Threshold():forward(conv2)
local max2 = nn.TemporalMaxPooling(3, 3):forward(thresh2)
-- 110 x 256
local conv3 = nn.TemporalConvolution(256, 256, 3):forward(max2)
local thresh3 = nn.Threshold():forward(conv3)
-- 108 x 256
local conv4 = nn.TemporalConvolution(256, 256, 3):forward(thresh3)
local thresh4 = nn.Threshold():forward(conv4)
-- 106 x 256
local conv5 = nn.TemporalConvolution(256, 256, 3):forward(thresh4)
local thresh5 = nn.Threshold():forward(conv5)
-- 104 x 256
local max3 = nn.TemporalMaxPooling(3, 3):forward(thresh5)
-- 16 x 256
local rs = nn.Reshape(4096):forward(max3)
-- 8192
local fc1 = nn.Linear(4096, 1024):forward(rs)
local drop = nn.Dropout(0.5):forward(fc1)
local fc2 = nn.Linear(1024, emb_dim):forward(drop)
]]

print(doc_enc)