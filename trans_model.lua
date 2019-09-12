require 'torch'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'
require 'nngraph'
require 'loadcaffe'
require 'lfs'
require 'io'
require 'image'
local utils = require 'model.lm.utils'
local net_utils = require 'model.lm.net_utils'
require 'model.cnn.CNN'

--------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

function LanguageModel_SC_CNN_PAD0_MEX_13_TO_LanguageModel_SC_LCNN()
  require 'model.lm.LanguageModel'
  -- target
  require 'model.lm.LanguageModel_SC_LCNN'
  -- source
  require 'model.lm.LanguageModel_SC_CNN_PAD0_MEX_13'
  in_model = 'checkpoint/tmp/model_id_SC_CNN_PAD0_MEX_1320161009_best_1009bak2.t7'
  local checkpoint = torch.load(in_model)
  local protos = checkpoint.protos
  in_lm = protos.lm

  local lmOpt = {}
  lmOpt.cell_size = protos.lm.cell_size
  lmOpt.vocab_size = protos.lm.vocab_size
  lmOpt.input_encoding_size = protos.lm.input_encoding_size
  lmOpt.rnn_size = protos.lm.rnn_size
  lmOpt.num_layers = protos.lm.num_layers
  lmOpt.core_width = protos.lm.core_width
  lmOpt.dropout = protos.lm.dropout
  lmOpt.seq_length = protos.lm.seq_length
  lmOpt.batch_size = protos.lm.batch_size

  out_lm = nn.LanguageModel_SC_LCNN(lmOpt)
  -- init cnn
  -- init lstm
  out_lm.core = in_lm.core
  out_lm.char_cnn = in_lm.char_cnn
  -- init lookuptable
  out_lm.lookup_table.weight = torch.Tensor(out_lm.lookup_table.weight:size()):copy(in_lm.lookup_table.weight)
  out_lm.lookup_table_tc.weight = torch.Tensor(out_lm.lookup_table_tc.weight:size()):copy(in_lm.lookup_table_tc.weight)
  local lm_modules = out_lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end

  torch.save('out.t7', out_lm)
end

function LanguageModel_CNN_PAD0_RNN_TO_LanguageModel_SC_LCNN()
  require 'model.lm.LanguageModel'
  -- target
  require 'model.lm.LanguageModel_SC_LCNN'
  -- source
  require 'model.lm.LanguageModel_SC_CNN_PAD0_MEX_13'
  in_model = 'checkpoint/tmp/model_id_SC_CNN_PAD0_MEX_1320161009_best_1009bak2.t7'
  local checkpoint = torch.load(in_model)
  local protos = checkpoint.protos
  in_lm = protos.lm

  local lmOpt = {}
  lmOpt.cell_size = protos.lm.cell_size
  lmOpt.vocab_size = protos.lm.vocab_size
  lmOpt.input_encoding_size = protos.lm.input_encoding_size
  lmOpt.rnn_size = protos.lm.rnn_size
  lmOpt.num_layers = protos.lm.num_layers
  lmOpt.core_width = protos.lm.core_width
  lmOpt.dropout = protos.lm.dropout
  lmOpt.seq_length = protos.lm.seq_length
  lmOpt.batch_size = protos.lm.batch_size

  out_lm = nn.LanguageModel_SC_LCNN(lmOpt)
  -- init cnn
  -- init lstm
  out_lm.core = in_lm.core
  out_lm.char_cnn = in_lm.char_cnn
  -- init lookuptable
  out_lm.lookup_table.weight = torch.Tensor(out_lm.lookup_table.weight:size()):copy(in_lm.lookup_table.weight)
  out_lm.lookup_table_tc.weight = torch.Tensor(out_lm.lookup_table_tc.weight:size()):copy(in_lm.lookup_table_tc.weight)
  local lm_modules = out_lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end

  torch.save('out.t7', out_lm)
end

LanguageModel_CNN_PAD0_RNN_TO_LanguageModel_SC_LCNN()
