require 'nn'
local utils = require 'model.lm.utils'
local net_utils = require 'model.lm.net_utils'

local LanguageCNN = require 'model.lm.LanguageCNN'
-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel_GCNN', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.core_width = utils.getopt(opt, 'core_width', 16)
  self.batch_size = utils.getopt(opt, 'batch_size', 1)
  -- create the core cnn network. note +1 for both the START and END tokens
  self.core = LanguageCNN.LanguageCNN_Gated()
  self.start_lookup_table_from = utils.getopt(opt, 'start_lookup_table_from','')
  if string.len(self.start_lookup_table_from) >0 then
    self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
    self.lookup_table.weight = torch.Tensor(self.lookup_table.weight:size()):copy(lm_init.lookup_table.weight)
  else
    self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
  end
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  -- construct the initial state for the LSTM
  self.init_state1 = torch.CudaTensor(batch_size, 12, 512):fill(0)
  self.init_state2 = torch.CudaTensor(batch_size, 8, 512):fill(0)
  self.init_state3 = torch.CudaTensor(batch_size, 6, 512):fill(0)
  self.init_state4 = torch.CudaTensor(batch_size, 4, 512):fill(0)
  self.init_state5 = torch.CudaTensor(batch_size, 2, 512):fill(0)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  --self.init_state = torch.CudaTensor(batch_size, 512):fill(0)
  self.num_state = 1
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  for t=2,self.seq_length+2 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
  end
end

function layer:getModulesList()
  return {self.core, self.lookup_table}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M,
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(imgs, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end -- indirection for beam search

  local batch_size = imgs:size(1)
  self:_createInitState(batch_size)
  local state1 = self.init_state1
  local state2 = self.init_state2
  local state3 = self.init_state3
  local state4 = self.init_state4
  local state5 = self.init_state5

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step
  local cat_tmp
  local concat_all
  local lookup_table_in = {}
  local lookup_table_out = {}
  for t=1,self.seq_length+2 do
    local concat_1, concat_2, concat_3, concat_4, concat_5, concat_6
    local xt, it, sampleLogprobs, concat_temp, text_condition, image_temp
    if t == 1 then
      -- feed in the images
      xt = torch.CudaTensor(imgs:size()[1],self.core_width*imgs:size()[2])
      for i=1,self.core_width do
        cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(imgs)
      end
      concat_temp = imgs
    elseif t == 2 then
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      lookup_table_in[t] = it
      concat_temp = self.lookup_table:forward(it)
      lookup_table_out[t] = concat_temp
      local imgs_zero = torch.CudaTensor(imgs:size()):fill(0)
      xt = torch.CudaTensor(imgs:size()[1],self.core_width*imgs:size()[2])
      for i=1,self.core_width do
        if i < t then cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(lookup_table_out[t+1-i])
        else cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(imgs_zero)
        end
      end
    else
      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end
      lookup_table_in[t] = it
      concat_temp = self.lookup_table:forward(it)
      lookup_table_out[t] = concat_temp
      xt = torch.CudaTensor(imgs:size()[1],self.core_width*imgs:size()[2])
      local imgs_zero = torch.CudaTensor(imgs:size()):fill(0)
      for i=1,self.core_width do
        if i < t then cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(lookup_table_out[t+1-i])
        else cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(imgs_zero)
        end
      end
    end

    if t >= 3 then
      seq[t-2] = it -- record the samples
      seqLogprobs[t-2] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end
    --local inputs = {xt, concat_temp, imgs, state}
    local inputs = {xt, state1, state2, state3, state4, state5}
    local outs = self.core:forward(inputs) -- softmax output is the output vector
    logprobs = outs[1]
    state1 = outs[2]
    state2 = outs[3]
    state3 = outs[4]
    state4 = outs[5]
    state5 = outs[6]
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs, core_out
end
--https://github.com/Cloud-CV/diverse-beam-search/blob/master/eval.lua
function layer:sample_beam_new(imgs, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local div_group = utils.getopt(opt,'div_group', 4)
  local lambda = utils.getopt(opt,'lambda',0.4)
  local batch_size, feat_dim = imgs:size(1), imgs:size(2)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  -- lets process every image independently for now, for simplicity
  for k=1,batch_size do
    bdash = beam_size/div_group -- bdash beams per group
    -- TABLE OF LSTM STATES
    state_table = {}
    for i=1,div_group do state_table[i] = {} end
    beam_seq_table = {} -- table that stores the sequences
    for i=1,div_group do beam_seq_table[i] = torch.LongTensor(self.seq_length, bdash):zero() end
    beam_seq_logprobs_table = {} --table to store logprobs of sequences
    for i=1,div_group do beam_seq_logprobs_table[i] = torch.LongTensor(self.seq_length, bdash):zero() end
    beam_logprobs_sum_table = {} --table to store joint probability
    for i=1,div_group do beam_logprobs_sum_table[i] = torch.CudaTensor(bdash):fill(0) end
    -- logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    done_beams_table = {} -- done beams!
    for i=1,div_group do done_beams_table[i] = {} end
    done_beams_flag = {}
    for i=1,div_group do done_beams_flag[i] = torch.CudaTensor(bdash):fill(0) end
    state = {} -- init state with zeros
    state = torch.CudaTensor(bdash, 512):fill(0)
    for i=1,div_group do state_table[i] = state:clone() end
    logprobs_table = {}
    for i=1,div_group do logprobs_table[i] = torch.zeros(bdash, self.vocab_size) end
    --print('done with inits')
    local lookup_table_in = {}
    local lookup_table_out = {}
    local imgk = imgs[{ {k,k} }]:expand(bdash, feat_dim) -- k'th image feature expanded out
    for t=1,self.seq_length+div_group+1 do
      for divm = 1,div_group do
        local xt, it, sampleLogprobs, concat_2, concat_3, concat_temp, text_condition, image_temp
        if t >= divm and t<= self.seq_length+divm+1 then
          -- feed in the images
          if t==divm then
            xt = torch.CudaTensor(imgk:size()[1],self.core_width*imgk:size()[2])
            for i=1,self.core_width do
              cat_tmp = xt:sub(1,imgk:size()[1],(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(imgk)
            end
            concat_temp = imgk
          elseif t == divm+1 then
            -- feed in the start tokens
            it = torch.LongTensor(bdash):fill(self.vocab_size+1)
            lookup_table_in[t] = it
            concat_temp = self.lookup_table:forward(it)
            lookup_table_out[t] = concat_temp

            local imgs_zero = torch.CudaTensor(imgk:size()):fill(0)
            xt = torch.CudaTensor(imgk:size()[1],self.core_width*imgk:size()[2])
            for i=1,self.core_width do
              if i < t then cat_tmp = xt:sub(1,imgk:size()[1],(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(lookup_table_out[t+1-i])
              else cat_tmp = xt:sub(1,imgk:size()[1],(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(imgs_zero)
              end
            end
          else
            --[[
            perform a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences.
            ]]--
            local logprobsf = logprobs_table[divm] -- lets go to CPU for more efficiency in indexing operations
            if divm > 1 then
              time_slice = t- divm -1
              for prev_choice =1,divm-1 do
                prev_decisions = beam_seq_table[prev_choice][time_slice]
                for sub_beam =1,bdash do
                  for prev_labels =1,bdash do
                    logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - lambda
                  end
                end
              end
            end
            beam_seq_table,beam_seq_logprobs_table,beam_logprobs_sum_table,state_table,done_beams_table,done_beams_flag = net_utils.beam_step_cnn(logprobsf,bdash,t,divm,self.vocab_size,self.seq_length,beam_seq_table,beam_seq_logprobs_table,beam_logprobs_sum_table,state_table,done_beams_table,done_beams_flag)
            --print(beam_seq_logprobs_table)
            -- encode as vectors
            --print('time: '..t..' divm: ' .. divm..'\n')
            it = beam_seq_table[divm][t-divm-1]
            lookup_table_in[t] = it
            concat_temp = self.lookup_table:forward(it)
            lookup_table_out[t] = concat_temp

            local row_size = it:size(1)
            xt = torch.CudaTensor(row_size,self.core_width*imgk:size()[2])
            local imgs_zero = torch.CudaTensor(imgk:size()):fill(0)
            for i=1,self.core_width do
              if i < t then cat_tmp = xt:sub(1,row_size,(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(lookup_table_out[t+1-i])
              else cat_tmp = xt:sub(1,row_size,(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(imgs_zero)
              end
            end
          end
          local inputs = {xt, concat_temp, imgk, state_table[divm]}
          local outs = self.core:forward(inputs) -- softmax output is the output vector
          logprobs_table[divm] = outs[1]:clone() -- last element is the output vector
          temp_state=outs[2]
          state_table[divm]=temp_state:clone()
        end
      end
    end
    final_beams = {}
    --print(done_beams_table)
    for group_index = 1,div_group do
      for sub_beam = 1,bdash do
        temp_index = (group_index-1)*bdash+sub_beam
        final_beams[temp_index] = done_beams_table[group_index][sub_beam]
      end
    end
    table.sort(final_beams, compare)
    seq[{ {}, k }] = final_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = final_beams[1].logps
  end
  --print(final_beams)
  -- return the samples and their log likelihoods
  return seq, seqLogprobs, final_beams
end

--[[
Implements beam search. Really tricky indexing stuff going on inside.
Not 100% sure it's correct, and hard to fully unit test to satisfaction, but
it seems to work, doesn't crash, gives expected looking outputs, and seems to
improve performance, so I am declaring this correct.
]]--
function layer:sample_beam(imgs, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 10)
  local batch_size, feat_dim = imgs:size(1), imgs:size(2)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  -- lets process every image independently for now, for simplicity
  for k=1,batch_size do

    -- create initial states for all beams
    self:_createInitState(beam_size)
    local state = self.init_state

    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    local imgk = imgs[{ {k,k} }]:expand(beam_size, feat_dim) -- k'th image feature expanded out
    local lookup_table_in = {}
    local lookup_table_out = {}

    for t=1,self.seq_length+2 do
      local xt, it, sampleLogprobs, concat_2, concat_3, concat_temp, text_condition, image_temp
      local new_state
      if t == 1 then
        -- feed in the images
        xt = torch.CudaTensor(imgk:size()[1],self.core_width*imgk:size()[2])
        for i=1,self.core_width do
          cat_tmp = xt:sub(1,imgk:size()[1],(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(imgk)
        end
        concat_temp = imgk
      elseif t == 2 then
        -- feed in the start tokens
        it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
        lookup_table_in[t] = it
        concat_temp = self.lookup_table:forward(it)
        lookup_table_out[t] = concat_temp
        local imgs_zero = torch.CudaTensor(imgk:size()):fill(0)
        xt = torch.CudaTensor(imgk:size()[1],self.core_width*imgk:size()[2])
        for i=1,self.core_width do
          if i < t then cat_tmp = xt:sub(1,imgk:size()[1],(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(lookup_table_out[t+1-i])
          else cat_tmp = xt:sub(1,imgk:size()[1],(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(imgs_zero)
          end
        end
      else
        --[[
        perform a beam merge. that is,
        for every previous beam we now many new possibilities to branch out
        we need to resort our beams to maintain the loop invariant of keeping
        the top beam_size most likely sequences.
        ]]--
        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 3 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        --new_state = net_utils.clone_list(state)
        new_state = state:clone()
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 3 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-3}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-3}, {} }]:clone()
        end
        for vix=1,beam_size do
          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 3 then
            beam_seq[{ {1,t-3}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-3}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          -- copy over state in previous beam q to new beam at vix
          -- copy over state in previous beam q to new beam at vix
          new_state[vix] = state[v.q]
          -- append new end terminal at the end of this beam
          beam_seq[{ t-2, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-2, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

          if v.c == self.vocab_size+1 or t == self.seq_length+2 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(),
                logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                p = beam_logprobs_sum[vix]
              })
          end
        end

        -- encode as vectors
        it = beam_seq[t-2]
        lookup_table_in[t] = it
        concat_temp = self.lookup_table:forward(it)
        lookup_table_out[t] = concat_temp

        local row_size = it:size(1)
        xt = torch.CudaTensor(row_size,self.core_width*imgk:size()[2])
        local imgs_zero = torch.CudaTensor(imgk:size()):fill(0)
        for i=1,self.core_width do
          if i < t then cat_tmp = xt:sub(1,row_size,(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(lookup_table_out[t+1-i])
          else cat_tmp = xt:sub(1,row_size,(i-1)*imgk:size()[2]+1,i*imgk:size()[2]):copy(imgs_zero)
          end
        end
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams
      local inputs = {xt, concat_temp, imgk, state}
      local outs = self.core:forward(inputs) -- softmax output is the output vector
      logprobs = outs[1]
      state = outs[2]
    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end
--[[
input is a tuple of:
1. torch.Tensor of size NxK (K is dim of image code)
2. torch.LongTensor of size DxN, elements 1..M
where M = opt.vocab_size and D = opt.seq_length

returns a (D+2)xNx(M+1) Tensor giving (normalized) log probabilities for the
next token at every iteration of the LSTM (+2 because +1 for first dummy
  img forward, and another +1 because of START/END tokens shift)
--]]
function layer:updateOutput(input)
  local imgs = input[1]
  local seq = input[2]
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+2, batch_size, self.vocab_size+1)

  self:_createInitState(batch_size)

  self.state1 = {[0] = self.init_state1}
  self.state2 = {[0] = self.init_state2}
  self.state3 = {[0] = self.init_state3}
  self.state4 = {[0] = self.init_state4}
  self.state5 = {[0] = self.init_state5}

  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.lookup_tables_outputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  for t=1,self.seq_length+2 do

    local can_skip = false
    local xt, concat_temp, text_condition, image_temp
    if t == 1 then
      -- feed in the images
      xt = torch.CudaTensor(imgs:size()[1],self.core_width*imgs:size()[2])
      for i=1,self.core_width do
        cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(imgs)
      end
      concat_temp = imgs
    elseif t == 2 then
      -- feed in the start tokens
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      concat_temp = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
      self.lookup_tables_outputs[t] = concat_temp

      xt = torch.CudaTensor(imgs:size()[1],self.core_width*imgs:size()[2])
      local imgs_zero = torch.CudaTensor(imgs:size()):fill(0)
      for i=1,self.core_width do
        if i < t then cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(self.lookup_tables_outputs[t+1-i])
        else cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(imgs_zero)
        end
      end
    else
      -- feed in the rest of the sequence...
      local it = seq[t-2]:clone()
      if torch.sum(it) == 0 then
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true
      end
      --[[
      seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
      that won't make lookup_table crash with an error.
      token #1 will do, arbitrarily. This will be ignored anyway
      because we will carefully set the loss to zero at these places
      in the criterion, so computation based on this value will be noop for the optimization.
      --]]
      it[torch.eq(it,0)] = 1

      if not can_skip then
        self.lookup_tables_inputs[t] = it
        concat_temp = self.lookup_tables[t]:forward(it)
        self.lookup_tables_outputs[t] = concat_temp
        xt = torch.CudaTensor(imgs:size()[1],self.core_width*imgs:size()[2])
        local imgs_zero = torch.CudaTensor(imgs:size()):fill(0)
        for i=1,self.core_width do
          if i < t then cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(self.lookup_tables_outputs[t+1-i])
          else cat_tmp = xt:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(imgs_zero)
          end
        end
      end
    end

    if not can_skip then
      -- construct the inputs
      --self.inputs[t] = {xt, concat_temp, imgs, self.state[t-1]}
      self.inputs[t] = {xt, self.state1[t-1], self.state2[t-1],self.state3[t-1],self.state4[t-1],self.state5[t-1]}
      -- forward the network
      local out = self.clones[t]:forward(self.inputs[t])
      self.output[t] = out[1]
      self.state1[t] = out[2]
      self.state2[t] = out[3]
      self.state3[t] = out[4]
      self.state4[t] = out[5]
      self.state5[t] = out[6]
      self.tmax = t
    end
  end

  return self.output
end

--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
  local dimgs -- grad on input images

  -- go backwards and lets compute gradients
  local imgs = input[1]
  local dstate1 = {[self.tmax] = self.init_state1}
  local dstate2 = {[self.tmax] = self.init_state2}
  local dstate3 = {[self.tmax] = self.init_state3}
  local dstate4 = {[self.tmax] = self.init_state4}
  local dstate5 = {[self.tmax] = self.init_state5}
  local dprevOut
  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    table.insert(dout, gradOutput[t])
    table.insert(dout, dstate1[t])
    table.insert(dout, dstate2[t])
    table.insert(dout, dstate3[t])
    table.insert(dout, dstate4[t])
    table.insert(dout, dstate5[t])
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    local dxt = dinputs[1]
    dstate1[t-1] = dinputs[2]
    dstate2[t-1] = dinputs[3]
    dstate3[t-1] = dinputs[4]
    dstate4[t-1] = dinputs[5]
    dstate5[t-1] = dinputs[6]
    if t == 1 then
      --dimgs:add(dinputs[3])
      dimgs = dinputs[1]:sub(1,imgs:size()[1],1,imgs:size()[2])
    else
      -- split the first gradient to next CNN
      dprevOut = dinputs[1]:sub(1,imgs:size()[1],1,imgs:size()[2])
      --dprevOut:add(dinputs[2])
      self.lookup_tables[t]:backward(self.lookup_tables_inputs[t], dprevOut) -- backprop into lookup table
      --[[
      if t == self.tmax then
        dimgs = dinputs[3]
      else
        dimgs:add(dinputs[3])
      end
      ]]
    end
  end

  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {dimgs, torch.Tensor()}
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion_GCNN', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

--[[
input is a Tensor of size (D+2)xNx(M+1)
seq is a LongTensor of size DxN. The way we infer the target
in this criterion is as follows:
- at first time step the output is ignored (loss = 0). It's the image tick
- the label sequence "seq" is shifted by one to produce targets
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=2,L do -- iterate over sequence time (ignore t=1, dummy forward for the image)

      -- fetch the index of the next token in the sequence
      local target_index
      if t-1 > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t-1,b}] -- t-1 is correct, since at t=2 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end

    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
