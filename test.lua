require 'torch'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'
require 'nngraph'
require 'loadcaffe'
require 'lfs'
require 'io'
require 'xlua' 
require 'image'
local utils = require 'model.lm.utils'
require 'model.cnn.CNN'
require 'model.lm.DataLoader'
require 'model.lm.DataLoaderRaw'
require 'model.lm.LanguageModel'
require 'model.gan.txt2image'
--------------------------------------------------------------------------------
require 'model.lm.LanguageModel'
--------------------------------------------------------------------------------
local net_utils = require 'model.lm.net_utils'
local html = '<html>' ..
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
cmd:option('-dataset', 'mscoco','the dataset name ')
cmd:option('-rnn_type','NORNN','rnn type')
cmd:option('-lcnn_type', 'LCNN_16IN_4Layer_MaxPool','the cnn model name ')
cmd:option('-model', 'checkpoint/CNN-VGG16_LCNN-LCNN_16IN-RNN-RHN_best_score.t7','path to model to evaluate')
cmd:option('-wordembedding', 'w2v','use w2v')
cmd:option('-semantic_words', false,'use semantic_words')
cmd:option('-att_type','','rnn type')
cmd:option('-input_h5', 'data/coco/cocotalk_0113.h5','path to the h5file containing the preprocessed dataset.')
cmd:option('-input_json', 'data/coco/cocotalk_0113.json','path to the json file containing additional info and vocab. ')
cmd:option('-att_type','','attention type')
cmd:option('-image_mapping', 1,'add image mapping?')
cmd:option('-core_mask', 16, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-seq_length', 16, 'max. length of a sentence')
cmd:option('-batch_size', 20, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-dump_images', 1, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_path', 1, 'Write image paths along with predictions into vis json? (1=yes,0=no)')
cmd:option('-dump_dir', 'results', 'Write image paths along with predictions into vis json? (1=yes,0=no)')
cmd:option('-dump_html', true, 'Output detected results to html file ? (true=yes,false=no)')
cmd:option('-verbose', true, 'Print image paths along with predictions ? (true=yes,false=no)')
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
cmd:option('-image_folder', '', 'If this is nonempty then will predict on the images in this folder path')
cmd:option('-image_root', '', 'In case the image paths have to be preprended with a root path to an image folder')
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
cmd:option('-coco_json', '', 'if nonempty then use this file in DataLoaderRaw (see docs there).')
cmd:option('-coco_challenge', 0, 'generate challenge json files')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-fixgpu', 1, 'fix gpu id')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')

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
if opt.semantic_words == 1 and opt.dataset == 'mscoco' then 
  opt.image_mapping = 1 
  opt.att_type = ''
  opt.input_h5 = 'data/coco/cocotalk_semantic_words.h5'
  opt.input_json = 'data/coco/cocotalk_semantic_words.json'
elseif opt.semantic_words == 0 and opt.dataset == 'mscoco' then 
  opt.image_mapping = 0
  opt.input_h5 = 'data/coco/cocotalk_0113.h5'
  opt.input_json = 'data/coco/cocotalk_0113.json'
elseif opt.semantic_words == 0 and opt.dataset == 'flickr30K' then 
  opt.image_mapping = 0
  opt.input_h5 = 'data/flickr30k/data_flickr30k.h5'
  opt.input_json = 'data/flickr30k/data_flickr30k.json'
end

--[[
If run for challenge, load from raw file and disable language evaluation
]]
total_num = 0
if opt.split == 'test' and opt.coco_challenge == 1 then
  opt.image_folder = 'data/coco/test2014/'
  opt.coco_json = 'data/coco/annotations/image_info_test2014.json'
  opt.language_eval = 0
  total_num = 40775
elseif opt.split == 'val' and opt.coco_challenge == 1 then
  opt.image_folder = 'data/coco/val2014/'
  opt.coco_json = 'data/coco/annotations/captions_val2014.json'
  opt.language_eval = 0
  total_num = 40504
else
  total_num = 5000
end

local model_name = 'LCNN-'.. opt.lcnn_type .. '_RNN-' .. opt.rnn_type .. ''
-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = { 'rnn_size','input_encoding_size','drop_prob_lm','cnn_proto','cnn_model','seq_per_img'}
for k,v in pairs(fetch) do
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
local vocab = checkpoint.vocab -- ix -> word mapping

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader
if string.len(opt.image_folder) == 0 then
  loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
else
  loader = DataLoaderRaw{ folder_path = opt.image_folder, coco_json = opt.coco_json}
end

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos

protos.crit = nn.LanguageModelCriterion()
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.lm:createClones() -- reconstruct clones inside the language model
for k,v in pairs(protos) do v:cuda() end

-------------------------------------------------------------------------------
-- get catural generated image captions
-------------------------------------------------------------------------------
local tg = {}
local cg = {}
local cg2 = {}
local conv1 = {}
local function fetch_gencap_conv(n_index_cp, tg, cg, cg2, seq_len)
  n_index_cp = n_index_cp + 1
  tg[n_index_cp] = torch.FloatTensor(17):zero()
  cg[n_index_cp] = torch.FloatTensor(17):zero()
  cg2[n_index_cp] = torch.FloatTensor(17):zero()
  --conv1 = torch.FloatTensor(12,12):zero()

  for i=1,seq_len do
    tg[n_index_cp][i] = core_out[1][i+1]
    cg[n_index_cp][i] = core_out[2][i+1]
    cg2[n_index_cp][i] = core_out[3][i+1]
  end
  return n_index_cp, tg, cg, cg2
end

local function write_gencap(entry, seq_len, n_index_cp)
  local conv1 = {}
  if seq_len == 12 then
    local file = io.open("exchange.txt", "w")
    file:write(entry.caption)
    file:close()
    for i=1,12 do
      conv1[i] = torch.mean(conv1_out[i]:float(),3)
    end
  end
  return conv1
end
-------------------------------------------------------------------------------
-- Evaluation function
-------------------------------------------------------------------------------
local function eval_split(split, evalopt, model_name)
  local verbose = utils.getopt(evalopt, 'verbose', false)
  local dump_html = utils.getopt(evalopt, 'dump_html', false)
  local num_images = utils.getopt(evalopt, 'num_images', true)
  local total_num = utils.getopt(evalopt, 'total_num', true)

  protos.cnn:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local n_index = 0
  local n_index_cp = 0

  ------------------------------------------------------------------------------
  if dump_html then
    local out_html_name = opt.dataset .. '_' .. model_name
    fname_html = string.format('results/caption_%s.html', out_html_name)
    os.execute(string.format('rm -rf %s', fname_html))
    os.execute(string.format('echo "%s" >> %s', html, fname_html))
  end
  ------------------------------------------------------------------------------
  while true do
    local tmp_html = ''
    -- fetch a batch of data
    local data = loader:getBatch{ batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img}
    local raw_images = data.images
    -- preprocess in place, and don't augment
    data.images = net_utils.prepro('VGG16', data.images, false, opt.gpuid >= 0)
    n = n + data.images:size(1)
    xlua.progress(n, total_num)
    local feats 
    local exp_attrs
    if opt.semantic_words then -- expand the data.semantic_words: batch_size * 16(attribute words per image)
      exp_attrs = protos.expander:forward(data.semantic_words):clone()
    end
    feats = protos.cnn:forward(data.images)
    local expanded_feats = protos.expander:forward(feats)
    -- evaluate loss if we have the labels
    local logprobs, attrs, attrs_alpha
    local loss = 0
    if data.labels then
      if opt.semantic_words then
        logprobs, attrs, attrs_alpha = protos.lm:forward{expanded_feats, data.labels, exp_attrs}
      else
        logprobs = protos.lm:forward{expanded_feats, data.labels}
      end
      loss = protos.crit:forward(logprobs, data.labels)
      loss_sum = loss_sum + loss
      loss_evals = loss_evals + 1
    end
    local seq
    -- forward the model to also get generated samples for each image
    local sample_opts = { sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }

    if opt.semantic_words then
      seq = protos.lm:sample({feats, data.semantic_words})
    else
      seq = protos.lm:sample({feats},sample_opts)
    end

    local sents = net_utils.decode_sequence(vocab, seq)
    local _,seq_len = string.gsub(sents[1], "%S+", "")
    --[[
    n_index_cp, tg, cg, cg2 = fetch_gencap_conv(n_index_cp, tg, cg, cg2, seq_len)
    ]]
    local raw_sents = net_utils.decode_sequence(vocab, data.labels)
    -- output captions to file or print them !!!

    for k=1,#sents do
      n_index = n_index + 1
      local entry = {image_id = data.infos[k].id, caption = sents[k]}
      if opt.dump_path == 1 then entry.file_name = data.infos[k].file_path end
      table.insert(predictions, entry)
      if opt.dump_images == 1 then
        -- dump the raw image to vis/ folder
        local cur_dir = lfs.currentdir ()
        local save_raw_im_path = path.join(opt.dump_dir, opt.dataset .. '/' .. data.infos[k].file_path)
        image.save(save_raw_im_path, raw_images[k])
        --local cmd = os.execute('cp "' .. path.join(opt.image_root, data.infos[k].file_path) .. '" vis/imgs/img' .. #predictions .. '.jpg' )
      end
      local ground_truth = ''
      local image_name
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
        --[[
        local conv1 = write_gencap(entry, seq_len, n_index_cp)
        ]]
        image_name = path.join(opt.image_root, data.infos[k].file_path)
        for i=1,5 do
          ground_truth = ground_truth .. string.format('%s</br>', raw_sents[(k-1)*5+i])
        end
      end
      if dump_html then
        tmp_html = tmp_html .. string.format('\n<tr><td><img src="%s" width="100"></td><td>%d</td><td>%d</td><td><b><font style="color:red">%s</font></b></td><td>%s</td></tr>', image_name, n_index, entry.image_id, entry.caption, ground_truth)
      end
    end
    -- write html string to file
    if dump_html then
      os.execute(string.format('echo "%s" >> %s', tmp_html, fname_html))
    end
    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, num_images)
    if verbose then print(string.format('evaluating performance... %d/%d (%f)', ix0-1, ix1, loss)) end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if num_images >= 0 and n >= num_images then break end -- we've used enough images
  end

  -- disp progress
  xlua.progress(n, num_images)
  -- end the html file
  if dump_html then
    os.execute(string.format('echo "%s" >> %s', '</html>', fname_html))
  end

  local lang_stats
  if opt.coco_challenge == 0 and opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id, opt.dataset)
  end

  if n % opt.batch_size * 100 == 0 then collectgarbage() end

  return loss_sum/loss_evals, predictions, lang_stats
end

local loss, split_predictions, lang_stats = eval_split(opt.split, {num_images = opt.num_images, total_num = total_num, verbose = opt.verbose, dump_html = opt.dump_html}, model_name)
print('loss: ', loss)
if lang_stats then print(lang_stats)end


if opt.coco_challenge then
  local output_challenge = {}
  for i, entry in pairs(split_predictions) do
    table.insert(output_challenge, {image_id = tonumber(entry.image_id), caption = entry.caption}) 
  end
  local output_challenge_filename = ''
  if opt.split == 'val' then
    output_challenge_filename = 'captions_val2014_' .. model_name .. '_results.json'
  elseif opt.split == 'test' then
    output_challenge_filename = 'captions_test2014_'.. model_name .. '_results.json'
  end
  utils.write_json(output_challenge_filename, output_challenge)
  print('Save done in ' .. output_challenge_filename)
end

if opt.dump_json == 1 then
  -- dump the json
  utils.write_json('vis/vis.json', split_predictions)
end
