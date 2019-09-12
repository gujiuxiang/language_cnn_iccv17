require 'nn'
local utils = require 'model.lm.utils'
local net_utils = require 'model.lm.net_utils'

local LSTM = require 'model.rnn.LSTM'
local GRU = require 'model.rnn.GRU'
local RNN = require 'model.rnn.RNN'
local RHN = require 'model.rnn.RHN'
local NORNN = require 'model.rnn.NORNN'

require 'model.lm.LookupTableMaskZero'

local LanguageCNN = require 'model.lm.LanguageCNN'
-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.image_encoding_size = utils.getopt(opt, 'image_encoding_size')
  self.dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.rnn_type = utils.getopt(opt, 'rnn_type', 'GRU')
  self.rnn_size = utils.getopt(opt, 'rnn_size',512)
  self.att_type = utils.getopt(opt, 'att_type')
  self.att_size = utils.getopt(opt, 'attention_size', 128)
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.core_width = utils.getopt(opt, 'core_width', 16)
  self.batch_size = utils.getopt(opt, 'batch_size', 1)
  self.core_mask = utils.getopt(opt, 'core_mask', 16)

  self.image_mapping = utils.getopt(opt, 'image_mapping', 0)
  if self.image_mapping == 1 then
    self.linear_model = nn.Linear(self.image_encoding_size, self.rnn_size) -- for very first time step, just a linear layer
  end

  -- create the core cnn network. note +1 for both the START and END tokens
  self.lcnn_type = utils.getopt(opt, 'lcnn_type')
  if self.lcnn_type ~= '' then
    if self.core_width == 16 then
      if self.lcnn_type == 'LCNN_16IN' then self.lcnn = LanguageCNN.LanguageCNN_16In() 
      elseif self.lcnn_type == 'LCNN_16IN_4Layer' then self.lcnn = LanguageCNN.LanguageCNN_16In_4Layer() 
      elseif self.lcnn_type == 'LCNN_16IN_4Layer_MaxPool' then self.lcnn = LanguageCNN.LanguageCNN_16In_4Layer_MaxPool() 
      elseif self.lcnn_type == 'LCNN_16IN_3Layer' then self.lcnn = LanguageCNN.LanguageCNN_16In_3Layer() 
      elseif self.lcnn_type == 'LCNN_16IN_3Layer_MaxPool' then self.lcnn = LanguageCNN.LanguageCNN_16In_3Layer_MaxPool() 
      elseif self.lcnn_type == 'LCNN_16IN_NoIm' then self.lcnn = LanguageCNN.LanguageCNN_16In_NoIm() 
      elseif self.lcnn_type == 'LCNN_16IN_33355' then self.lcnn = LanguageCNN.LanguageCNN_16In_33355() 
      end
    elseif self.core_width == 8 then
      self.lcnn = LanguageCNN.LanguageCNN_8In() 
    elseif self.core_width == 4 then
      self.lcnn = LanguageCNN.LanguageCNN_4In() 
    end
    self.rnn_input_size = self.input_encoding_size*2
  else
    self.rnn_input_size = self.input_encoding_size
  end

  local rnnopt = {}
  rnnopt.word_embed_size = self.rnn_input_size
  rnnopt.input_size = self.rnn_input_size
  rnnopt.rnn_size = self.rnn_size
  rnnopt.output_size = self.vocab_size + 1
  rnnopt.num_layers = self.num_layers
  rnnopt.dropout = self.dropout

  if self.rnn_type == 'LSTM' then
    self.core = LSTM.lstm(rnnopt)
    self.state_num = 2
  elseif self.rnn_type == 'GRU' then
    self.core = GRU.gru(rnnopt)
    self.state_num = 1
  elseif self.rnn_type == 'RHN' then
    self.core = RHN.rhn(rnnopt)
    self.state_num = 1
  elseif self.rnn_type == 'RNN' then
    self.core = RNN.rnn(rnnopt)
    self.state_num = 1
  elseif self.rnn_type == 'NORNN' then
    self.core = NORNN.rnn(rnnopt)
    self.state_num = 1
  elseif self.rnn_type == 'NORNN_ADD' then
    self.core = NORNN.rnn_add(rnnopt)
    self.state_num = 1
  elseif self.rnn_type == 'LSTM_GATT' then
    self.input_attention_model = LSTM.Make_Input_Attention_Bilinear_Unit(self.input_encoding_size, self.input_encoding_size, self.rnn_size)
    self.core= LSTM.with_output_attention(rnnopt)
    self.state_num = 2
  elseif self.rnn_type == 'GRU_GATT' then
    self.input_attention_model = GRU.Make_Input_Attention_Bilinear_Unit(self.input_encoding_size, self.input_encoding_size, self.rnn_size)
    self.core= GRU.with_output_attention(rnnopt)
    self.state_num = 1
  elseif self.rnn_type == 'RHN_GATT' then
    self.input_attention_model = RHN.Make_Input_Attention_Bilinear_Unit(self.input_encoding_size, self.input_encoding_size, self.rnn_size)
    self.core= RHN.with_output_attention(rnnopt)
    self.state_num = 1
  elseif self.rnn_type == 'RNN_GATT' then
    self.input_attention_model = RNN.Make_Input_Attention_Bilinear_Unit(self.input_encoding_size, self.input_encoding_size, self.rnn_size)
    self.core= RNN.with_output_attention(rnnopt)
    self.state_num = 1
  end
  -- create lookup table
  self.lookup_table = nn.LookupTableMaskZero(self.vocab_size + 1, self.input_encoding_size)

  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  -- cell numer, for lstm, hidden and cell; for gru/rnn/rhn, hidden.
  for h=1,self.num_layers*self.state_num do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  print('constructing clones inside the LanguageModel') -- construct the net clones
  if self.image_mapping == 1 then self.linear_models = {self.linear_model} end
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}

  if self.att_type ~= '' then self.input_attention_model_clones = {self.input_attention_model} end
  if self.lcnn_type ~= '' then self.lcnns = {self.lcnn} end

  for t=2,self.seq_length+2 do
    if self.att_type ~= '' then
      self.input_attention_model_clones[t] = self.input_attention_model:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    if self.lcnn_type ~= '' then
      self.lcnns[t] = self.lcnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
  end

  if self.att_type ~= '' then
    -- add an additional lookup_tables for attributes vector
    self.lookup_tables[#self.lookup_tables + 1] = self.lookup_table:clone('weight', 'gradWeight')
  end
end

function layer:getModulesList()
  if self.image_mapping == 1 then
    if self.lcnn_type ~= '' then return {self.linear_model, self.core, self.lcnn, self.lookup_table}
    elseif self.att_type ~= '' then return {self.linear_model, self.core, self.input_attention_model, self.lookup_table}
    else return {self.linear_model, self.core, self.lookup_table}
    end
  else
    if self.lcnn_type ~= '' then return {self.core, self.lcnn, self.lookup_table}
    elseif self.att_type ~= '' then return {self.core, self.input_attention_model, self.lookup_table}
    else return {self.core, self.lookup_table}
    end
  end
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()
  local p3,g3
  if self.image_mapping == 1 then p3,g3 = self.linear_model:parameters() end
  local p4,g4,p5,g5
  if self.lcnn_type ~= '' then p4, g4 = self.lcnn:parameters() end
  if self.att_type ~= '' then p5, g5 = self.input_attention_model:parameters() end

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  if self.image_mapping == 1 then for k,v in pairs(p3) do table.insert(params, v) end end
  if self.lcnn_type ~= '' then for k,v in pairs(p4) do table.insert(params, v) end end
  if self.att_type ~= '' then for k,v in pairs(p5) do table.insert(params, v) end end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  if self.image_mapping == 1 then for k,v in pairs(g3) do table.insert(grad_params, v) end end
  if self.lcnn_type ~= '' then for k,v in pairs(g4) do table.insert(grad_params, v) end end
  if self.att_type ~= '' then for k,v in pairs(g5) do table.insert(grad_params, v) end end
  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  if self.image_mapping == 1 then for k,v in pairs(self.linear_models) do v:training() end end
  if self.lcnn_type ~= '' then for k,v in pairs(self.lcnns) do v:training() end end
  if self.att_type ~= '' then for k,v in pairs(self.input_attention_model_clones) do v:training() end end
  for k,v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  if self.image_mapping == 1 then for k,v in pairs(self.linear_models) do v:evaluate() end end
  if self.lcnn_type ~= '' then for k,v in pairs(self.lcnns) do v:evaluate() end end
  if self.att_type ~= '' then for k,v in pairs(self.input_attention_model_clones) do v:evaluate() end end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
end

function layer:languageCNN_In_Pad_1st(row_size, imgs)
  xt = torch.CudaTensor(row_size,self.core_width*imgs:size()[2])
  for i=1,self.core_width do
    if self.core_mask == nil then self.core_mask = 16 end
    if i > self.core_mask then 
        break
    else
      cat_tmp = xt:sub(1,row_size,(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(imgs)
    end
  end
  return xt
end

function layer:languageCNN_In_Pad(t, row_size, imgs, lookup_table_out)
  xt = torch.CudaTensor(row_size,self.core_width*imgs:size()[2])
  local imgs_zero = torch.CudaTensor(imgs:size()):fill(0)
  for i=1,self.core_width do
      if self.core_mask == nil then self.core_mask = 16 end
      if i > self.core_mask then 
        break
      else
        if i < t then cat_tmp = xt:sub(1,row_size,(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(lookup_table_out[t+1-i])
        else cat_tmp = xt:sub(1,row_size,(i-1)*imgs:size()[2]+1,i*imgs:size()[2]):copy(imgs_zero)
        end
      end
  end
  return xt
end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M,
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(input, opt)
  local imgs_in = input[1]
  local attrs
  if self.att_type ~= nil then attrs = input[2] end

  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  if sample_max == 1 and beam_size > 1 then -- indirection for beam search
    if self.att_type ~= nil then return self:sample_beam({imgs_in, attrs}, opt)
    else return self:sample_beam({imgs_in}, opt) end
  end

  local batch_size = imgs_in:size(1)
  self:_createInitState(batch_size)
  local state = self.init_state

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step
  local cat_tmp
  local concat_all
  local lookup_table_in = {}
  local lookup_table_out = {}
  ---------------------------------------------------------------
  -- only used for debug
  local debug_flag = false
  local conv1_out = {}
  local conv2_out = {}
  local transform_gate = {}
  local carry_gate = {}
  local carry_gate2 = {}

  local conv1_out_mean= torch.FloatTensor(13, 12):zero()
  local conv2_out_mean= torch.FloatTensor(13, 8):zero()
  local conv3_out_mean= torch.FloatTensor(13, 6):zero()
  local conv4_out_mean= torch.FloatTensor(13, 4):zero()
  local conv5_out_mean= torch.FloatTensor(13, 2):zero()
  local convf_out_mean= torch.FloatTensor(13, 1):zero()
  local mout_mean= torch.FloatTensor(13, 1):zero()
  ---------------------------------------------------------------

  if self.att_type ~= '' then
    -- in the future, we will use an exclusive lookup_tables
    -- to embedding the attribute words
    self.As = self.lookup_table:forward(attrs):clone()
  end

  for t=1,self.seq_length+2 do
    local concat_1, concat_2, concat_3, concat_4, concat_5, concat_6
    local xt, it, sampleLogprobs, concat_temp, text_condition, image_temp
    if t == 1 then
      -- feed in the images
      if self.image_mapping == 1 then
        imgs = self.linear_model:forward(imgs_in)
      else
        imgs = imgs_in
      end
      if self.lcnn_type ~= '' then
        xt = self:languageCNN_In_Pad_1st(imgs:size()[1], imgs)
        concat_temp = imgs
      else
        xt = imgs
      end
    elseif t == 2 then
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      if self.lcnn_type ~= '' then
        lookup_table_in[t] = it
        concat_temp = self.lookup_table:forward(it)
        lookup_table_out[t] = concat_temp
        xt = self:languageCNN_In_Pad(t, imgs:size()[1], imgs, lookup_table_out)
      elseif self.att_type ~= '' then
        x_word = self.lookup_table:forward(it)
        xt = self.input_attention_model:forward({x_word, self.As})
      else
        xt = self.lookup_table:forward(it)
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
      if self.lcnn_type ~= '' then
        lookup_table_in[t] = it
        concat_temp = self.lookup_table:forward(it)
        lookup_table_out[t] = concat_temp
        xt = self:languageCNN_In_Pad(t, imgs:size()[1], imgs, lookup_table_out)
      elseif self.att_type ~= '' then
        x_word = self.lookup_table:forward(it)
        xt = self.input_attention_model:forward({x_word, self.As})
      else
        xt = self.lookup_table:forward(it)
      end
    end

    if t >= 3 then
      seq[t-2] = it -- record the samples
      seqLogprobs[t-2] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end
    local core_in
    if self.lcnn_type ~= '' then
      core_in = self.lcnn:forward({xt, concat_temp, imgs})
    else
      core_in = xt
    end
    local inputs
    if self.att_type ~= '' then
      inputs = {core_in, self.As, unpack(state)}
    elseif self.rnn_type == 'NORNN_ADD' then
      inputs = {core_in, unpack(state)}
    else
      inputs = {core_in, unpack(state)}
    end
    local out = self.core:forward(inputs) -- softmax output is the output vector
    logprobs = out[self.num_state+1] -- last element is the output vector
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end
    ---------------------------------------------------------------
    if debug_flag then
      conv1_out[t]=self.core.forwardnodes[8].data.module.output
      conv2_out[t]=self.core.forwardnodes[13].data.module.output

      local tmp_conv1 = torch.mean(self.core.forwardnodes[8].data.module.output,3)
      local tmp_conv2 = torch.mean(self.core.forwardnodes[13].data.module.output,3)
      local tmp_conv3 = torch.mean(self.core.forwardnodes[18].data.module.output,3)
      local tmp_conv4 = torch.mean(self.core.forwardnodes[23].data.module.output,3)
      local tmp_conv5 = torch.mean(self.core.forwardnodes[28].data.module.output,3)
      local tmp_convf = torch.mean(self.core.forwardnodes[48].data.module.output)
      local tmp_mconv = torch.mean(self.core.forwardnodes[50].data.module.output)
      if t>1 and t<=14 then
        for j=1,12 do conv1_out_mean[t-1][j]= tmp_conv1[1][j][1] end
        for j=1,8 do conv2_out_mean[t-1][j]= tmp_conv2[1][j][1] end
        for j=1,6 do conv3_out_mean[t-1][j]= tmp_conv3[1][j][1] end
        for j=1,4 do conv4_out_mean[t-1][j]= tmp_conv4[1][j][1] end
        for j=1,2 do conv5_out_mean[t-1][j]= tmp_conv5[1][j][1] end
        convf_out_mean[t-1]= tmp_convf
        mout_mean[t-1]= tmp_mconv
      end
    end
    ---------------------------------------------------------------
    if debug_flag then
      transform_gate[t] = torch.mean(self.core.forwardnodes[60].data.module.output)
      carry_gate[t] = torch.mean(self.core.forwardnodes[69].data.module.output)
      carry_gate2[t] = torch.mean(self.core.forwardnodes[50].data.module.output)
    end
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs, core_out
end

--https://github.com/Cloud-CV/diverse-beam-search/blob/master/eval.lua
function layer:sample_beam_new(input, opt)
  local imgs = input[1]
  local attrs
  if opt.att_type ~= nil then attrs = input[2] end

  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local div_group = utils.getopt(opt,'div_group', 4)
  local lambda = utils.getopt(opt,'lambda',0.4)
  local batch_size, feat_dim = imgs:size(1), imgs:size(2)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  -- lets process every image independently for now, for simplicity
  if self.att_type ~= '' then
    self.As = self.lookup_table:forward(attrs):clone()
  end

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
            xt = self:languageCNN_In_Pad_1st(imgk:size()[1], imgk)
            concat_temp = imgk
          elseif t == divm+1 then
            -- feed in the start tokens
            it = torch.LongTensor(bdash):fill(self.vocab_size+1)
            lookup_table_in[t] = it
            concat_temp = self.lookup_table:forward(it)
            lookup_table_out[t] = concat_temp
            xt = self:languageCNN_In_Pad(t, imgk:size()[1], imgk, lookup_table_out)
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
            xt = self:languageCNN_In_Pad(t, row_size, imgk, lookup_table_out)
          end
          local lcnn_out = self.lcnn:forward({xt, concat_temp, imgk})
          local inputs = {lcnn_out, state_table[divm]}

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
function layer:sample_beam(input, opt)
  local imgs_in = input[1]
  local attrs
  if opt.att_type ~= '' then attrs = input[2] end

  local feat_dim
  local beam_size = utils.getopt(opt, 'beam_size', 10)
  local batch_size = imgs_in:size(1)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  -- lets process every image independently for now, for simplicity
  if self.att_type ~= '' then
    self.As = self.lookup_table:forward(attrs):clone()
  end

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
    if self.image_mapping == 1 then
      imgs = self.linear_model:forward(imgs_in)
    else
      imgs = imgs_in
    end
    feat_dim = imgs:size(2)
    local imgk = imgs[{ {k,k} }]:expand(beam_size, feat_dim) -- k'th image feature expanded out
    local lookup_table_in = {}
    local lookup_table_out = {}
    local Ak
    if self.att_type ~= '' then
      Ak = self.As[{{k, k},{}, {}}]:expand(beam_size, self.As:size(2), self.As:size(3)) -- expand out Ak
    end
    for t=1,self.seq_length+2 do
      local x1, xt, it, sampleLogprobs, concat_2, concat_3, concat_temp, text_condition, image_temp
      local new_state
      if t == 1 then
        -- feed in the images
        if self.lcnn_type ~= '' then
          xt = self:languageCNN_In_Pad_1st(imgk:size()[1], imgk)
          concat_temp = imgk
        else
          xt = imgk
        end
      elseif t == 2 then
        -- feed in the start tokens
        it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
        if self.lcnn_type ~= '' then
          lookup_table_in[t] = it
          concat_temp = self.lookup_table:forward(it)
          lookup_table_out[t] = concat_temp
          xt = self:languageCNN_In_Pad(t, imgk:size()[1], imgk, lookup_table_out)
        elseif self.att_type ~= '' then
          x_word = self.lookup_table:forward(it)
          xt = self.input_attention_model:forward({x_word, Ak})
        else
          xt = self.lookup_table:forward(it)
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
        new_state = net_utils.clone_list(state)
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
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          -- copy over state in previous beam q to new beam at vix
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
        if self.lcnn_type ~= '' then
          lookup_table_in[t] = it
          concat_temp = self.lookup_table:forward(it)
          lookup_table_out[t] = concat_temp

          local row_size = it:size(1)
          xt = self:languageCNN_In_Pad(t, row_size, imgk, lookup_table_out)
        elseif self.att_type ~= '' then
          x_word = self.lookup_table:forward(it)
          xt = self.input_attention_model:forward({x_word, Ak})
        else
          xt = self.lookup_table:forward(it)
        end
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams
      local core_in
      if self.lcnn_type ~= '' then
        core_in = self.lcnn:forward({xt, concat_temp, imgk})
      else
        core_in = xt
      end
      local inputs
      if self.att_type ~= '' then
        inputs = {core_in, Ak, unpack(state)}
      elseif self.rnn_type == 'NORNN_ADD' then
        inputs = {core_in, unpack(state)}
      else
        inputs = {core_in, unpack(state)}
      end

      local out = self.core:forward(inputs) -- softmax output is the output vector
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
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
  local imgs_in = input[1] -- 10 * 11, bz * img_enc
  local seq = input[2] -- 7 * 10, seq_len * bz
  local attrs
  if self.att_type ~= '' then
    attrs = input[3] -- 10 * 10, bz * top_attr
  end

  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+2, batch_size, self.vocab_size+1)

  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}

  if self.att_type ~= '' then
    self.As = self.lookup_tables[#self.lookup_tables]:forward(attrs) --bz * top_attrs * 11, ie, 10x10x11
  end

  self.input_attention_in = {}
  self.lcnn_inputs = {}
  self.rnn_inputs = {}
  self.lookup_tables_inputs = {}
  self.lookup_tables_outputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  for t=1,self.seq_length+2 do

    local can_skip = false
    local x_word, xt, concat_temp, text_condition, image_temp
    if t == 1 then
      if self.image_mapping == 1 then
        imgs = self.linear_models[t]:forward(imgs_in) -- NxK sized input
      else
        imgs = imgs_in
      end
      -- feed in the images
      if self.lcnn_type ~= '' then
        xt = self:languageCNN_In_Pad_1st(imgs:size()[1], imgs)
        concat_temp = imgs
      else
        xt = imgs -- NxK sized input
      end
    elseif t == 2 then
      -- feed in the start tokens
      local it = torch.CudaTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      if self.lcnn_type ~= '' then
        concat_temp = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
        self.lookup_tables_outputs[t] = concat_temp
        xt= self:languageCNN_In_Pad(t, imgs:size()[1], imgs, self.lookup_tables_outputs)
      elseif self.att_type ~= '' then
        x_word = self.lookup_tables[t]:forward(it)
        self.input_attention_in[t] = {x_word, self.As}
        xt = self.input_attention_model_clones[t]:forward({x_word, self.As})
      else
        xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
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
        if self.lcnn_type ~= '' then
          concat_temp = self.lookup_tables[t]:forward(it)
          self.lookup_tables_outputs[t] = concat_temp
          xt= self:languageCNN_In_Pad(t, imgs:size()[1], imgs, self.lookup_tables_outputs)
        elseif self.att_type ~= '' then
          x_word = self.lookup_tables[t]:forward(it)
          self.input_attention_in[t] = {x_word, self.As}
          xt = self.input_attention_model_clones[t]:forward({x_word, self.As})
        else
          xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
        end
      end
    end

    if not can_skip then
      -- construct the inputs
      local core_in
      if self.lcnn_type ~= '' then
        self.lcnn_inputs[t] = {xt, concat_temp, imgs}
        core_in = self.lcnns[t]:forward(self.lcnn_inputs[t])
      else
        core_in = xt
      end
      if self.att_type ~= '' then
        self.rnn_inputs[t] = {core_in, self.As, unpack(self.state[t-1])}
      elseif self.rnn_type == 'NORNN_ADD' then
        self.rnn_inputs[t] = {core_in, unpack(self.state[t-1])}
      else
        self.rnn_inputs[t] = {core_in, unpack(self.state[t-1])}
      end
      -- forward the network
      local out = self.clones[t]:forward(self.rnn_inputs[t])
      self.output[t] = out[self.num_state+1] -- last element is the output vector
      self.state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
      self.tmax = t
    end
  end

  return self.output
end

--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
  --local dimgs = nil -- grad on input images
  local dimgs = torch.Tensor(input[1]:size()):type(input[1]:type()):zero() -- grad on input images
  local dAs = nil -- grad on the attrs

  -- go backwards and lets compute gradients
  local imgs = input[1]
  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  local dprevOut
  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    local dxt
    dstate[t-1] = {} -- copy over rest to state grad
    -----------------------------------------------------------------------------------------------
    if self.lcnn_type ~= '' then
      local drnn_inputs = self.clones[t]:backward(self.rnn_inputs[t], dout)
      if self.rnn_type == 'NORNN_ADD' then
        for k=2,self.num_state+1 do table.insert(dstate[t-1], drnn_inputs[k]) end
      else
        for k=2,self.num_state+1 do table.insert(dstate[t-1], drnn_inputs[k]) end
      end

      local dlcnn_inputs = self.lcnns[t]:backward(self.lcnn_inputs[t], drnn_inputs[1])
      dxt = dlcnn_inputs[1]

      if t == 1 then
        -- need to backprop t times
        for i=1,self.core_width do
          dimgs:add(dlcnn_inputs[1]:sub(1,imgs:size()[1],(i-1)*imgs:size()[2]+1,i*imgs:size()[2]))
        end
        dimgs:add(dlcnn_inputs[3])
        if self.image_mapping == 1 then
          dimgs_in = self.linear_models[t]:backward(input[1], dimgs)
        else
          dimgs_in = dimgs
        end
      else
        -- split the first gradient to next CNN
        dprevOut = dlcnn_inputs[1]:sub(1,imgs:size()[1],1,imgs:size()[2]) -- gradients of the current word from lcnn
        dprevOut:add(dlcnn_inputs[2]) -- gradients of current word
        self.lookup_tables[t]:backward(self.lookup_tables_inputs[t], dprevOut) -- backprop into lookup table

        dimgs:add(dlcnn_inputs[3]) -- gradients from lcnn
      end
    -----------------------------------------------------------------------------------------------
    elseif self.att_type ~= '' then
      local dinputs = self.clones[t]:backward(self.rnn_inputs[t], dout)
      -- split the gradient to dg_t and to state
      local dg_t = dinputs[1]
      if dAs == nil then
        dAs = torch.Tensor():typeAs(dinputs[2]):resizeAs(dinputs[2]):copy(dinputs[2])
      else
        dAs = dAs + dinputs[2]
      end

      for k = 3, self.num_state + 2 do table.insert(dstate[t-1], dinputs[k]) end

      -- continue backprop of xt
      if t == 1 then
        -- self.glimpses[1] is output of self.linear_model with input as imgs
        dimgs = self.linear_models[t]: backward(input[1], dg_t)
      else
        -- backward to self.input_attention_models
        local dattention_out = self.input_attention_model_clones[t]:backward(self.input_attention_in[t], dg_t)

        local dword_t, dAs_t2 = dattention_out[1], dattention_out[2]
        local it = self.lookup_tables_inputs[t]
        self.lookup_tables[t]:backward(it, dword_t) -- backprop into lookup table
        dAs = dAs + dAs_t2
      end
    -----------------------------------------------------------------------------------------------
    else
      local dinputs = self.clones[t]:backward(self.rnn_inputs[t], dout)
      dxt = dinputs[1] -- first element is the input vector
      for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end

      -- continue backprop of xt
      if t == 1 then
        dimgs = dxt
        dimgs_in = self.linear_models[t]:backward(input[1], dimgs)
      else
        local it = self.lookup_tables_inputs[t]
        self.lookup_tables[t]:backward(it, dxt) -- backprop into lookup table
      end
    end
    -----------------------------------------------------------------------------------------------
  end

  -- do we need to update gradients to self.lookup_tabls[1] using dAs? currently, let us say 'no' for simplicity
  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  if self.att_type ~= '' then
    self.lookup_tables[#self.lookup_tables]:backward(input[3],dAs)
    self.gradInput = {dimgs_in, torch.Tensor(), torch.Tensor()}
  else
    self.gradInput = {dimgs_in, torch.Tensor()}
  end
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
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

function crit:precision(input, seq)
  --input: predicted sentence (word sequence)
  --seq: ground-truth sequence of fword ids
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-2, 'input Tensor should be 2 larger in time')

  -- hit count for opt.seq_length and <EOS> token
  local hit_count = torch.FloatTensor(D+1):fill(0)
  --local hit_count = torch.FloatTensor(D):fill(0)
  local perplexity = 0
  local precision = 0
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

      -- if there is a non-null next token, enforce precision
      if target_index ~= 0 then
        local output_cpu = input[{t,b,{}}]:float():exp()
        local prob, predicted_word_label
        prob, predicted_word_label = output_cpu:max(1)
        if predicted_word_label[1] == Mp1 then break end
        if predicted_word_label[1] == target_index then
          -- accumulate 
          hit_count[t-1] = hit_count[t-1] + 1
          perplexity = perplexity - math.log(prob:squeeze())
        end
      end
    end
    n = n + 1
  end
  -- nomalize by number of predictions that were made
  precision = hit_count:div(n)
  perplexity = math.pow(2.0, perplexity / n )
  return precision, perplexity
end
