require 'nn'
require 'cunn'
require 'model.lm.CompactBilinearPooling'
local cbptest = torch.TestSuite()
local precision = 1e-5
local debug = false

function testPsi()
   local homogeneous = true
   local batch = 1
   local dim = 5
   local S = 100
   local x = torch.rand(batch,dim):cuda()
   local y = torch.rand(batch,dim):cuda()
   local tmp = 0
   local c
   for s=1,S do
      c = nn.CompactBilinearPooling(dim, homogeneous):cuda()
      c:forward({x,y})
      tmp = tmp + c.y[1][1]:dot(c.y[2][1])
   end
   local xy = x[1]:dot(y[1])
   local diff = math.abs(tmp/S - xy)
   assert(diff / xy < .1, 'error_ratio='..diff / xy..', E[<phi(x,h,s),phi(y,h,s)>]=<x,y>')
end

testPsi()