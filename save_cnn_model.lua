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
require 'model.cnn.CNN'
require 'model.lm.DataLoader'
require 'model.lm.DataLoaderRaw'
require 'model.lm.LanguageModel'
require 'model.gan.txt2image'
--------------------------------------------------------------------------------
-- (MS COCO) Image CNN + Langugage Model (LSTM)
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel'
-- (MS COCO) Image CNN + SC Langugage Model (LSTM)
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_BNLSTM_SC'
-- (MS COCO) Image CNN + Langugage Model (RHN)
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_RHW'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_GRU'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_RNN'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_GRU_NEW'

-- (MS COCO) Image CNN + Language CNN + LSTM
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_SC_CNN_PAD0_MEX_13'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_SC_CNN_PAD0_MEX_SCLSTM2'
-- (MS COCO) Image CNN + Language CNN + RHN
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_16IN_PAD0_5RHW_NEW'
-- (flickr30K) Image CNN + Language CNN + GRU
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_GRU_30K'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_16IN_PAD0_RNN'

-- (flickr30K) Image CNN + Language CNN + LSTM (2 layers)
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_SC_CNN_PAD0_MEX_SCLSTM2_30K'
-- (flickr30K) Image CNN + Language CNN + RHN
require 'model.lm.LanguageModel_CNN_16IN_PAD0_1RHW_30K'
--require 'model.lm.LanguageModel_CNN_16IN_PAD0_5RHW_NEW_30K'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_16IN_PAD0_RHN_GS'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_16IN_PAD0_RHN_GS2'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_16IN_PAD0_RHN_GS3'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_16IN_PAD0_RHN_GS4'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_16IN_PAD0_RHN_GS6'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_16IN_PAD0_GRU_COCO'
-- (flickr30K) Image CNN + Language CNN + GRU
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_30K'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_RHW_30K'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_RNN_30K'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_GRU_30K'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_GRU_30K'
require 'model.lm.bak.bak.2017-02-14-src.LanguageModel_CNN_RNN_30K'
--------------------------------------------------------------------------------
local net_utils = require 'model.lm.net_utils'
local html =  '<html>' ..
                '<body>' ..
                '<h1>Image Captions and Reconstructed Images</h1>' ..
                '<table border="1px solid gray" style="width=100%">'..
                  '<tr>'..
                    '<td><b>Image</b></td>'..
                    '<td><b>ID</b></td>'..
                    '<td><b>Image ID</b></td>'..
                    '<td><b>Generated Caption</b></td>'..
                    '<td><b>Ground truth</b></td>'..
                  '</tr>'

--------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')
local dataset = 'mscoco'
cmd:option('-dataset', 'flickr30K','the dataset name ')
cmd:option('-model', '/home/jxgu/github/cvpr2017_im2text_jxgu.old/torch7/checkpoint/tmp/model_id_basic20160828.t7','path to model to evaluate')
cmd:option('-model_name', 'LSTM','the language model name ')
cmd:option('-batch_size', 50, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-dump_images', 0, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_path', 1, 'Write image paths along with predictions into vis json? (1=yes,0=no)')
cmd:option('-dump_dir', 'results', 'Write image paths along with predictions into vis json? (1=yes,0=no)')
cmd:option('-dump_html', true, 'Output detected results to html file ? (true=yes,false=no)')
cmd:option('-verbose', true, 'Print image paths along with predictions ? (true=yes,false=no)')
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
cmd:option('-image_folder', '', 'If this is nonempty then will predict on the images in this folder path')
cmd:option('-image_root', '/home/jxgu/github/cvpr2017_im2text_jxgu/results/coco', 'In case the image paths have to be preprended with a root path to an image folder')
if dataset == 'flickr30K' then
  cmd:option('-input_h5', 'data/flickr30k/data_flickr30k.h5','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
  cmd:option('-input_json', 'data/flickr30k/data_flickr30k.json','path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
else
  cmd:option('-input_h5', 'data/coco/cocotalk_0113.h5','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
  cmd:option('-input_json', 'data/coco/cocotalk_0113.json','path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
end
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
cmd:option('-coco_json', '', 'if nonempty then use this file in DataLoaderRaw (see docs there).')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-fixgpu', 1, 'fix gpu id')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')


-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
cutorch.manualSeed(opt.seed)
if opt.fixegpu then
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

function replaceModules(net, orig_class_name, replacer)
  local nodes, container_nodes = net:findModules(orig_class_name)
  for i = 1, #nodes do
    for j = 1, #(container_nodes[i].modules) do
      if container_nodes[i].modules[j] == nodes[i] then
        local orig_mod = container_nodes[i].modules[j]
        print('replacing a cudnn module with nn equivalent...')
        print(orig_mod)
        container_nodes[i].modules[j] = replacer(orig_mod)
      end
    end
  end
end

function cudnnNetToCpu(net)
  local net_cpu = net:clone():float()
  replaceModules(net_cpu, 'cudnn.SpatialConvolution', 
    function(orig_mod)
      local cpu_mod = nn.SpatialConvolution(orig_mod.nInputPlane, orig_mod.nOutputPlane,
          orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH, orig_mod.padW, orig_mod.padH)
      cpu_mod.weight:copy(orig_mod.weight)
      cpu_mod.bias:copy(orig_mod.bias)
      cpu_mod.gradWeight = nil -- sanitize for thinner checkpoint
      cpu_mod.gradBias = nil -- sanitize for thinner checkpoint
      return cpu_mod
    end)
  replaceModules(net_cpu, 'cudnn.SpatialMaxPooling', 
    function(orig_mod)
      local cpu_mod = nn.SpatialMaxPooling(orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH, 
                                           orig_mod.padW, orig_mod.padH)
      return cpu_mod
    end)
  replaceModules(net_cpu, 'cudnn.ReLU', function() return nn.ReLU() end)
  return net_cpu
end
-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
local lstmcnn = checkpoint.protos.cnn
cpu_model = cudnnNetToCpu(lstmcnn)
net_utils.unsanitize_gradients(cpu_model)

torch.save('model/cnn/cnn_512_lstm_finetuned.t7', cpu_model)
collectgarbage()
