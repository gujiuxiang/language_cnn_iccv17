require 'lfs'
local utils = require 'model.lm.utils'
local net_utils = {}
local nninit = require 'nninit'
-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)

  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_part:add(layer)
  end

  cnn_part:add(nn.Linear(4096,encoding_size):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
  cnn_part:add(backend.ReLU(true))
  return cnn_part
end

-- Load resnet from facebook
function net_utils.build_cnn_resnet(cnn, opt)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  
  cnn_part = cnn
  cnn_part:remove(#cnn_part.modules)

  cnn_part:add(nn.Linear(2048,encoding_size))
  cnn_part:add(nn.ReLU(true))
  
  if backend == 'cudnn' then
    cudnn.convert(cnn_part, cudnn)
  end
  return cnn_part
end

-- takes a batch of images and preprocesses them
-- VGG-16 network is hardcoded, as is 224 as size to forward
function net_utils.prepro(cnn_type, imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')
  -------------------------------------------------------------------------------------
  -------------------------------------------------------------------------------------
  if cnn_type == 'VGG16' then
    local h,w = imgs:size(3), imgs:size(4)
    local cnn_input_size = 224

    -- cropping data augmentation, if needed
    if h > cnn_input_size or w > cnn_input_size then
      local xoff, yoff
      if data_augment then
        xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
      else
        -- sample the center
        xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
      end
      -- crop.
      imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
    end

    -- ship to gpu or convert from byte to float
    if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

    -- lazily instantiate vgg_mean
    if not net_utils.vgg_mean then
      net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
    end
    net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match

    -- subtract vgg mean
    imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))
    return imgs
  -------------------------------------------------------------------------------------
  -------------------------------------------------------------------------------------
else
    fb_transforms = require 'model.lm.fb_transforms'
    imgs = imgs:float() 
    imgs:div(255)
    local meanstd = {
       mean = { 0.485, 0.456, 0.406 },
       std = { 0.229, 0.224, 0.225 },
    }

    local transform = fb_transforms.Compose{
       fb_transforms.ColorNormalize(meanstd),
       fb_transforms.CenterCrop(224),
    }
    
    local cnn_input_size = 224
    imgs_out = torch.Tensor(imgs:size(1), imgs:size(2), cnn_input_size, cnn_input_size):type(imgs:type())
    for i = 1, imgs:size(1) do
      imgs_out[i] = transform(imgs[i])
    end

    -- ship to gpu or convert from byte to float
    if on_gpu then imgs_out = imgs_out:cuda() else imgs_out = imgs_out:float() end
    return imgs_out
  end
  -------------------------------------------------------------------------------------
  -------------------------------------------------------------------------------------
end

function net_utils.build_cnn_seperate(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)

  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_part:add(layer)
  end

  local cnn_map_part = nn.Sequential()
  cnn_map_part:add(nn.Linear(4096,encoding_size):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
  cnn_map_part:add(backend.ReLU(true))
  return cnn_part, cnn_map_part
end

function net_utils.beam_step(logprobsf,beam_size,t,divm,vocab_size,seq_length,beam_seq_table,beam_seq_logprobs_table,beam_logprobs_sum_table,state_table,done_beams_table,done_beams_flag)
  --INPUTS:
  --logprobsf: probabilities augmented after diversity
  --beam_size: obvious
  --t        : time instant
  --beam_seq : tensor contanining the beams
  --beam_seq_logprobs: tensor contanining the beam logprobs
  --beam_logprobs_sum: tensor contanining joint logprobs
  --OUPUTS:
  --beam_seq
  --beam_seq_logprobs
  --beam_logprobs_sum
  --print('beam_size:' .. beam_size)
  local beam_seq = beam_seq_table[divm]
  local beam_seq_logprobs = beam_seq_logprobs_table[divm]
  local beam_logprobs_sum = beam_logprobs_sum_table[divm]
  t = t- divm -1
  local function compare(a,b) return a.p > b.p end -- used downstream
  ys,ix = torch.sort(logprobsf,2,true)
  candidates = {}
  cols = math.min(beam_size,ys:size()[2])
  rows = beam_size
  if t == 1 then rows = 1 end
  for c=1,cols do -- for each column (word, essentially)
    for q=1,rows do -- for each beam expansion
      --compute logprob of expanding beam q with word in (sorted) position c
      local local_logprob = ys[{ q,c }]
      local candidate_logprob = beam_logprobs_sum[q] + local_logprob
      table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
  end
  end
  table.sort(candidates, compare)
  new_state = net_utils.clone_list(state)
  --local beam_seq_prev, beam_seq_logprobs_prev
  if t > 1 then
    --we''ll need these as reference when we fork beams around
    beam_seq_prev = beam_seq[{ {1,t-1}, {} }]:clone()
    beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-1}, {} }]:clone()
  end
  for vix=1,beam_size do
    v = candidates[vix]
    --fork beam index q into index vix
    if t > 1 then
      beam_seq[{ {1,t-1}, vix }] = beam_seq_prev[{ {}, v.q }]
      beam_seq_logprobs[{ {1,t-1}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
    end
    --rearrange recurrent states
    for state_ix = 1,#new_state do
      --  copy over state in previous beam q to new beam at vix
      new_state[state_ix][vix] = state[state_ix][v.q]
    end
    --append new end terminal at the end of this beam
    beam_seq[{ t, vix }] = v.c -- c'th word is the continuation
    beam_seq_logprobs[{ t, vix }] = v.r -- the raw logprob here
    beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam
    if v.c == vocab_size+1 or t == seq_length then
      if done_beams_flag[divm][vix] == 0 then
        --print('finished with beam '..'time: '..t)
        --END token special case here, or we reached the end.
        -- add the beam to a set of done beams
        done_beams_table[divm][vix] = {seq = beam_seq_table[divm][{ {}, vix }]:clone(), logps = beam_seq_logprobs_table[divm][{ {}, vix }]:clone(),p = beam_logprobs_sum_table[divm][vix]}
        done_beams_flag[divm][vix] = 1
      end
    end
  end
  if new_state then state = new_state end
  --print('done with 1 beam_step\n')
  return beam_seq_table,beam_seq_logprobs_table,beam_logprobs_sum_table,state_table,done_beams_table,done_beams_flag
end

function net_utils.beam_step_cnn(logprobsf,beam_size,t,divm,vocab_size,seq_length,beam_seq_table,beam_seq_logprobs_table,beam_logprobs_sum_table,state_table,done_beams_table,done_beams_flag)
  --INPUTS:
  --logprobsf: probabilities augmented after diversity
  --beam_size: obvious
  --t        : time instant
  --beam_seq : tensor contanining the beams
  --beam_seq_logprobs: tensor contanining the beam logprobs
  --beam_logprobs_sum: tensor contanining joint logprobs
  --OUPUTS:
  --beam_seq
  --beam_seq_logprobs
  --beam_logprobs_sum
  --print('beam_size:' .. beam_size)
  local beam_seq = beam_seq_table[divm]
  local beam_seq_logprobs = beam_seq_logprobs_table[divm]
  local beam_logprobs_sum = beam_logprobs_sum_table[divm]
  t = t- divm -1
  local function compare(a,b) return a.p > b.p end -- used downstream
  ys,ix = torch.sort(logprobsf,2,true)
  candidates = {}
  cols = math.min(beam_size,ys:size()[2])
  rows = beam_size
  if t == 1 then rows = 1 end
  for c=1,cols do -- for each column (word, essentially)
    for q=1,rows do -- for each beam expansion
      --compute logprob of expanding beam q with word in (sorted) position c
      local local_logprob = ys[{ q,c }]
      local candidate_logprob = beam_logprobs_sum[q] + local_logprob
      table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
  end
  end
  table.sort(candidates, compare)
  --new_state = net_utils.clone_list(state)
  new_state = state:clone()
  --local beam_seq_prev, beam_seq_logprobs_prev
  if t > 1 then
    --we''ll need these as reference when we fork beams around
    beam_seq_prev = beam_seq[{ {1,t-1}, {} }]:clone()
    beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-1}, {} }]:clone()
  end
  for vix=1,beam_size do
    v = candidates[vix]
    --fork beam index q into index vix
    if t > 1 then
      beam_seq[{ {1,t-1}, vix }] = beam_seq_prev[{ {}, v.q }]
      beam_seq_logprobs[{ {1,t-1}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
    end
    --rearrange recurrent states
    --[[
	for state_ix = 1,#new_state do
--  copy over state in previous beam q to new beam at vix
      new_state[state_ix][vix] = state[state_ix][v.q]
    end]]
    new_state[vix] = state[v.q]
    --append new end terminal at the end of this beam
    beam_seq[{ t, vix }] = v.c -- c'th word is the continuation
    beam_seq_logprobs[{ t, vix }] = v.r -- the raw logprob here
    beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam
    if v.c == vocab_size+1 or t == seq_length then
      if done_beams_flag[divm][vix] == 0 then
        --print('finished with beam '..'time: '..t)
        --END token special case here, or we reached the end.
        -- add the beam to a set of done beams
        done_beams_table[divm][vix] = {seq = beam_seq_table[divm][{ {}, vix }]:clone(), logps = beam_seq_logprobs_table[divm][{ {}, vix }]:clone(),p = beam_logprobs_sum_table[divm][vix]}
        done_beams_flag[divm][vix] = 1
      end
    end
  end
  if new_state then state = new_state end
  --print('done with 1 beam_step\n')
  return beam_seq_table,beam_seq_logprobs_table,beam_logprobs_sum_table,state_table,done_beams_table,done_beams_flag
end
function net_utils.build_cnn_conv(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 30)
  local backend = utils.getopt(opt, 'backend', 'cudnn')

  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_part:add(layer)
  end

  cnn_part:add(cnn:get(32))
  return cnn_part
end

function net_utils.build_cnn19(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 44)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)

  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_part:add(layer)
  end

  cnn_part:add(nn.Linear(4096,encoding_size))
  cnn_part:add(backend.ReLU(true))
  return cnn_part
end

-- just return a linear transform model
-- thus we can use features which are already extracted
-- here, we suppose we extract vgg 4096 features
-- for googlenet, feature size will be 1024 size features
function net_utils.build_linear_trans(opt)
    local backend = utils.getopt(opt, 'backend', 'cudnn')
    local encoding_size = utils.getopt(opt, 'encoding_size', 512)

    if backend == 'cudnn' then
        require 'cudnn'
        backend = cudnn
    elseif backend == 'nn' then
        require 'nn'
        backend = nn
    else
        error(string.format('Unrecognized backend %s', backend))
    end

    local linear_transform = nn.Sequential()
    linear_transform:add(nn.Linear(4096, encoding_size))
    linear_transform:add(backend.ReLU(true))

    return linear_transform
end


-- load glove vector
-- Parses and loads the GloVe word vectors into a hash table:
--   glove_table['word'] = vector
function net_utils.load_glove(glove_path, inputDim)
    local glove_file = io.open(glove_path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read GloVe text file one line at a time, break at EOF
        local k = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if k == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    -- may just use
                    -- glove_table[word] = torch.zeros(inputDim)
                    glove_table[word] = torch.zeros(inputDim) -- padded with an extra dimention for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][k-1] = tonumber(entry)
            end
            k = k+1
        end
        line = glove_file:read("*l")
    end

    return glove_table
end

function net_utils.prepro_googlenetV3(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 299
  local input_sub = 128
  local input_scale = 0.0078125

  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end

  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  imgs:mul(255):add(-input_sub):mul(input_scale)
  return imgs
end

function net_utils.prepro_resnet152(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 224
  local t = require 'model.lm.transforms'
  -- The model was trained with this input normalization
  local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
  }
  local transform = t.Compose{
    t.ColorNormalize(meanstd),
  }

  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end

  -- ship to gpu or convert from byte to float
  for i=1, imgs:size(1) do
    imgs[{i}] = transform(imgs[{i}] )
  end

  imgs = imgs:cuda()

  return imgs
end

function net_utils.prepro_googlenetV3_nomean(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 299
  local input_sub = 128
  local input_scale = 0.0078125
  local input_dim = 299
  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end
  --imgs = image.scale(imgs, input_dim)
  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  imgs:mul(255):add(-input_sub):mul(input_scale)
  return imgs
end

function net_utils.prepro_448(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 448

  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end

  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  if not net_utils.vgg_mean then
    net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  end
  net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match

  -- subtract vgg mean
  imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))

  return imgs
end

function net_utils.pospro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 224
  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  local vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  vgg_mean = vgg_mean:typeAs(imgs) -- a noop if the types match
  vgg_mean:div(255)
  imgs:add(1, vgg_mean:expandAs(imgs))

  return imgs
end

-- load glove vector
-- Parses and loads the GloVe word vectors into a hash table:
--   glove_table['word'] = vector
function net_utils.load_glove(glove_path, inputDim)
    local glove_file = io.open(glove_path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read GloVe text file one line at a time, break at EOF
        local k = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if k == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    -- may just use
                    -- glove_table[word] = torch.zeros(inputDim)
                    glove_table[word] = torch.zeros(inputDim) -- padded with an extra dimention for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][k-1] = tonumber(entry)
            end
            k = k+1
        end
        line = glove_file:read("*l")
    end

    return glove_table
end

-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2)
  local d = input:size(2)
  self.output:resize(input:size(1)*self.n, d)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, d) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

-- layer that expands features map  out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.Feat2DExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency

  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 3)
  local nc, fd = input:size(2), input:size(3)
  self.output:resize(input:size(1)*self.n, nc, fd)

  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {}, {}}]:expand(self.n, nc, fd) -- copy over
  end
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local nc, fd = input:size(2), input:size(3)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
    local m = node.data.module
    if m then
      table.insert(omg, m)
    end
  end
  return omg
end
function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, id, dataset)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}

  utils.write_json('coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  if dataset == 'mscoco' then
    os.execute('./misc/call_python_caption_eval.sh val' .. id .. '.json' .. ' ' .. dataset) -- i'm dying over here
  end
  local result_struct = utils.read_json('coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval_30K(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('30k-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval_30k.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('30k-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end


function net_utils.language_eval003(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption003/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval003.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption003/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

function net_utils.language_eval002(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption002/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval002.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption002/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end


function net_utils.language_eval001(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption001/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval001.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption001/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

function net_utils.language_eval_30k(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('30k-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval_30k.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('30k-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

function net_utils.language_eval_8k(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('8k-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval_8k.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('8k-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end


function net_utils.combine_all_parameters(...)
  --[[ like module:getParameters, but operates on many modules ]]--

  -- get parameters
  local networks = {...}
  local parameters = {}
  local gradParameters = {}
  for i = 1, #networks do
    local net_params, net_grads = networks[i]:parameters()
    if net_params then
      for _, p in pairs(net_params) do
        parameters[#parameters + 1] = p
      end
      for _, g in pairs(net_grads) do
        gradParameters[#gradParameters + 1] = g
      end
    end
  end

  local function storageInSet(set, storage)
    local storageAndOffset = set[torch.pointer(storage)]
    if storageAndOffset == nil then
      return nil
    end
    local _, offset = unpack(storageAndOffset)
    return offset
  end

  -- this function flattens arbitrary lists of parameters,
  -- even complex shared ones
  local function flatten(parameters)
    if not parameters or #parameters == 0 then
      return torch.Tensor()
    end
    local Tensor = parameters[1].new

    local storages = {}
    local nParameters = 0
    for k = 1,#parameters do
      local storage = parameters[k]:storage()
      if not storageInSet(storages, storage) then
        storages[torch.pointer(storage)] = {storage, nParameters}
        nParameters = nParameters + storage:size()
      end
    end

    local flatParameters = Tensor(nParameters):fill(1)
    local flatStorage = flatParameters:storage()

    for k = 1,#parameters do
      local storageOffset = storageInSet(storages, parameters[k]:storage())
      parameters[k]:set(flatStorage,
        storageOffset + parameters[k]:storageOffset(),
        parameters[k]:size(),
        parameters[k]:stride())
      parameters[k]:zero()
    end

    local maskParameters=  flatParameters:float():clone()
    local cumSumOfHoles = flatParameters:float():cumsum(1)
    local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
    local flatUsedParameters = Tensor(nUsedParameters)
    local flatUsedStorage = flatUsedParameters:storage()

    for k = 1,#parameters do
      local offset = cumSumOfHoles[parameters[k]:storageOffset()]
      parameters[k]:set(flatUsedStorage,
        parameters[k]:storageOffset() - offset,
        parameters[k]:size(),
        parameters[k]:stride())
    end

    for _, storageAndOffset in pairs(storages) do
      local k, v = unpack(storageAndOffset)
      flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
    end

    if cumSumOfHoles:sum() == 0 then
      flatUsedParameters:copy(flatParameters)
    else
      local counter = 0
      for k = 1,flatParameters:nElement() do
        if maskParameters[k] == 0 then
          counter = counter + 1
          flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
        end
      end
      assert (counter == nUsedParameters)
    end
    return flatUsedParameters
  end

  -- flatten parameters and gradients
  local flatParameters = flatten(parameters)
  local flatGradParameters = flatten(gradParameters)

  -- return new flat vector that contains all discrete parameters
  return flatParameters, flatGradParameters
end




function net_utils.clone_many_times(net, T)
  local clones = {}

  local params, gradParams
  if net.parameters then
    params, gradParams = net:parameters()
    if params == nil then
      params = {}
    end
  end

  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end

  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)

  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    if net.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
        cloneParams[i]:set(params[i])
        cloneGradParams[i]:set(gradParams[i])
      end
      if paramsNoGrad then
        cloneParamsNoGrad = clone:parametersNoGrad()
        for i =1,#paramsNoGrad do
          cloneParamsNoGrad[i]:set(paramsNoGrad[i])
        end
      end
    end

    clones[t] = clone
    collectgarbage()
  end

  mem:close()
  return clones
end

function net_utils.clone_many_times_multiple_nngraph(net, T)
  local clones = {}


  local params_table = {}
  local gradParams_table = {}
  local paramsNoGrad_table = {}
  for i=1, table.getn(net) do
    if net[i].parameters then
      local params, gradParams = net[i]:parameters()
      if params == nil then
        params = {}
      end
      table.insert(params_table, params)
      table.insert(gradParams_table, gradParams)
    end
    if net[i].parametersNoGrad then
      paramsNoGrad = net:parametersNoGrad()
      table.insert(paramsNoGrad_table, paramsNoGrad)
    end
  end


  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)

  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    for idx=1, table.getn(net) do
      local curr_net = net[idx]
      if curr_net.parameters then
        local cloneParams, cloneGradParams = clone[idx]:parameters()
        local cloneParamsNoGrad
        for i = 1, #params_table[idx] do
          cloneParams[i]:set(params_table[idx][i])
          cloneGradParams[i]:set(gradParams_table[idx][i])
        end
        if paramsNoGrad_table[idx] then
          cloneParamsNoGrad = clone[idx]:parametersNoGrad()
          for i =1,#paramsNoGrad[idx] do
            cloneParamsNoGrad[i]:set(paramsNoGrad_table[idx][i])
          end
        end
      end

    end
    clones[t] = clone
    collectgarbage()
  end

  mem:close()
  return clones
end

return net_utils
