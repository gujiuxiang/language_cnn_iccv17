--[[ 

Generic training script for GAN, GAN-CLS, GAN-INT, GAN-CLS-INT.

--]]
require 'image'
require 'graph'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'torch'
require 'optim'


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

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

txtSize = 1024 -- #  of dim for raw text.
local nc = 3
local nt = 128 -- #  of dim for text features.
local nz = 100 -- #  of dim for Z
local ngf = 196 -- #  of gen filters in first conv layer
local ndf = 196
local large = 0

----------------------------------------------------------------------------
local fcG = nn.Sequential()
fcG:add(nn.Linear(txtSize, nt))
fcG:add(nn.LeakyReLU(0.2,true))
local netG = nn.Sequential()

-- concat Z and txt
local ptg = nn.ParallelTable()
ptg:add(nn.Identity())
ptg:add(fcG)
netG:add(ptg)
netG:add(nn.JoinTable(2))

-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz + nt, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8))

-- state size: (ngf*8) x 4 x 4
local conc = nn.ConcatTable()
local conv = nn.Sequential()
conv:add(SpatialConvolution(ngf * 8, ngf * 2, 1, 1, 1, 1, 0, 0))
conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
conv:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
conv:add(SpatialConvolution(ngf * 2, ngf * 8, 3, 3, 1, 1, 1, 1))
conv:add(SpatialBatchNormalization(ngf * 8))
conc:add(nn.Identity())
conc:add(conv)
netG:add(conc)
netG:add(nn.CAddTable())

if large == 1 then
  -- state size: (ngf*8) x 4 x 4
  local conc = nn.ConcatTable()
  local conv = nn.Sequential()
  conv:add(SpatialConvolution(ngf * 8, ngf * 2, 1, 1, 1, 1, 0, 0))
  conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
  conv:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
  conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
  conv:add(SpatialConvolution(ngf * 2, ngf * 8, 3, 3, 1, 1, 1, 1))
  conv:add(SpatialBatchNormalization(ngf * 8))
  conc:add(nn.Identity())
  conc:add(conv)
  netG:add(conc)
  netG:add(nn.CAddTable())
end

netG:add(nn.ReLU(true))

-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4))

-- state size: (ngf*4) x 8 x 8
local conc = nn.ConcatTable()
local conv = nn.Sequential()
conv:add(SpatialConvolution(ngf * 4, ngf, 1, 1, 1, 1, 0, 0))
conv:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
conv:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
conv:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
conv:add(SpatialConvolution(ngf, ngf * 4, 3, 3, 1, 1, 1, 1))
conv:add(SpatialBatchNormalization(ngf * 4))
conc:add(nn.Identity())
conc:add(conv)
netG:add(conc)
netG:add(nn.CAddTable())

if large == 1 then
  -- state size: (ngf*4) x 8 x 8
  local conc = nn.ConcatTable()
  local conv = nn.Sequential()
  conv:add(SpatialConvolution(ngf * 4, ngf, 1, 1, 1, 1, 0, 0))
  conv:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
  conv:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
  conv:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
  conv:add(SpatialConvolution(ngf, ngf * 4, 3, 3, 1, 1, 1, 1))
  conv:add(SpatialBatchNormalization(ngf * 4))
  conc:add(nn.Identity())
  conc:add(conv)
  netG:add(conc)
  netG:add(nn.CAddTable())
end

netG:add(nn.ReLU(true))

-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2))
netG:add(nn.ReLU(true))

-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())

-- state size: (nc) x 64 x 64
netG:apply(weights_init)

----------------------------------------------------------------------------
convD = nn.Sequential()
-- input is (nc) x 64 x 64
convD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
convD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
convD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
convD:add(SpatialBatchNormalization(ndf * 2))
convD:add(nn.LeakyReLU(0.2, true))

-- state size: (ndf*2) x 16 x 16
convD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
convD:add(SpatialBatchNormalization(ndf * 4))
convD:add(nn.LeakyReLU(0.2, true))

-- state size: (ndf*4) x 8 x 8
convD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
convD:add(SpatialBatchNormalization(ndf * 8))

-- state size: (ndf*8) x 4 x 4
local conc = nn.ConcatTable()
local conv = nn.Sequential()
conv:add(SpatialConvolution(ndf * 8, ndf * 2, 1, 1, 1, 1, 0, 0))
conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
conv:add(SpatialConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 1, 1))
conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
conv:add(SpatialConvolution(ndf * 2, ndf * 8, 3, 3, 1, 1, 1, 1))
conv:add(SpatialBatchNormalization(ndf * 8))
conc:add(nn.Identity())
conc:add(skipD)
convD:add(conc)
convD:add(nn.CAddTable())

convD:add(nn.LeakyReLU(0.2, true))

local fcD = nn.Sequential()
fcD:add(nn.Linear(txtSize, nt))
fcD:add(nn.LeakyReLU(0.2,true))
fcD:add(nn.Replicate(4,3))
fcD:add(nn.Replicate(4,4)) 
local netD = nn.Sequential()
pt = nn.ParallelTable()
pt:add(convD)
pt:add(fcD)
netD:add(pt)
netD:add(nn.JoinTable(2))
-- state size: (ndf*8 + 128) x 4 x 4
netD:add(SpatialConvolution(ndf * 8 + nt, ndf * 8, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1
netD:apply(weights_init)
return netG, netD
