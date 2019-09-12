require 'paths'
require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'xlua' 

--cudnn.benchmark = true
--cudnn.fastest = true
cudnn.verbose = false

local utils = require 'misc.utils'
require 'model.lm.DataLoader'
require 'model.lm.DataLoaderRaw'
require 'model.lm.LanguageModel'
local net_utils = require 'model.lm.net_utils'

local model_filename = 'checkpoint/bak/LCNN-LCNN_16IN-RNN-LSTM_best_score.t7'
local seq_length = 16
local batch_size = 1

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Testing an Image Captioning model')
cmd:text()
cmd:text('Options')

cmd:option('-dataset', 'mscoco','the dataset name ')
cmd:option('-rnn_type','RHN','rnn type')
cmd:option('-lcnn_type', 'LCNN_16IN','the cnn model name ')

-- Input paths
cmd:option('-model', model_filename, 'path to model to evaluate')
cmd:option('-batch_size', 20, 'if > 0 then overrule, otherwise load from checkpoint.')
-- Basic options
cmd:option('-seq_length', seq_length, 'max. length of a sentence')
cmd:option('-batch_size', batch_size, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-dump_images', 0, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_json', 0, 'Dump json with predictions into vis folder? (1=yes,0=no)')
cmd:option('-dump_path', 0, 'Write image paths along with predictions into vis json? (1=yes,0=no)')
cmd:option('-verbose', false, 'Print image paths along with predictions ? (true=yes,false=no)')
-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

-- For evaluation on a folder of images:
cmd:option('-language_eval', 
  1,
  'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-image_folder', 
  --'',
  '/home/jxgu/github/cvpr2017_im2text_jxgu/torch7/data/coco/test2014/', 
  --'/home/jxgu/github/cvpr2017_im2text_jxgu/torch7/data/coco/val2014/', 
  'If this is nonempty then will predict on the images in this folder path')
cmd:option('-image_root', '', 'In case the image paths have to be preprended with a root path to an image folder')
-- For evaluation on MSCOCO images from some split:
cmd:option('-input_h5',
  '',
  'path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
cmd:option('-input_json',
  '',
  'path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
cmd:option('-split', 
  --'test', 
  'val', 
  'if running on MSCOCO images, which split to use: val|test|train')
cmd:option('-coco_json', 
  --'/home/jxgu/github/cvpr2017_im2text_jxgu/torch7/data/coco/annotations/captions_val2014.json',
  '/home/jxgu/github/cvpr2017_im2text_jxgu/torch7/data/coco/annotations/image_info_test2014.json',
  'if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
cmd:option('-coco_challenge', 
  true, 
  --true, 
  '')
-- misc
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()


local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
cutorch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed

local model_name = ''
batch_size = opt.batch_size
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
opt.seq_per_img = checkpoint.opt.seq_per_img
local vocab = checkpoint.vocab 
--print(vocab)


local loader
if string.len(opt.image_folder) == 0 then
  loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
else
  loader = DataLoaderRaw{folder_path = opt.image_folder, coco_json = opt.coco_json}
end


local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.crit = nn.LanguageModelCriterion()
protos.lm:createClones()
for k,v in pairs(protos) do v:cuda() end

protos.cnn:evaluate()
protos.lm:evaluate()
collectgarbage()

local total_size =0
if opt.split == 'test' then
  total_size = 40775
else
  total_size = 40504
end

local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', false)
  local num_images = utils.getopt(evalopt, 'num_images', true)

  -- rewind iteator back to first datapoint in the split
  loader:resetIterator(split)
  local n = 0
  local loss_sum = 0
  local accuracy = 0
  local perplexity = 0
  local loss_evals = 0
  local predictions = {}
  while true do
    -- fetch a batch of data
    
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img}
    if not opt.coco_challenge then data.labels = data.labels[{{1,opt.seq_length},{}}] end
    data.images = net_utils.prepro('VGG16', data.images, false, opt.gpuid >= 0)
    n = n + data.images:size(1)
    xlua.progress(n, total_size)
    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    -- evaluate loss if we have the labels
    local loss = 0
    if data.labels then
      local expanded_feats = protos.expander:forward(feats)
      local logprobs = protos.lm:forward{expanded_feats, data.labels}
      loss = protos.crit:forward(logprobs, data.labels)
      loss_sum = loss_sum + loss
    end

    -- forward the model to also get generated samples for each image
    local sample_opts = {sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }
    local seq = protos.lm:sample({feats}, sample_opts)
    local sents, num_words = net_utils.decode_sequence(vocab, seq)

    if not opt.coco_challenge then
      local gt_sents = ''
      for i=1,opt.seq_length do
        if vocab[tostring(data.labels[i][1])] ~= nil then
          gt_sents = 
            string.format('%s %s', gt_sents, vocab[tostring(data.labels[i][1])])
        end
      end
    end

    for k=1,#sents do
      local entry = {
        image_id = data.infos[k].id, 
        file_path = data.infos[k].file_path, 
        caption = sents[k], 
        gt = gt_sents}
      if opt.dump_path == 1 then
        entry.file_name = data.infos[k].file_path
      end
      table.insert(predictions, entry)
      if opt.dump_images == 1 then
        -- dump the raw image to vis/ folder
        local cmd = 'cp "' .. path.join(opt.image_root, data.infos[k].file_path) .. '" vis/imgs/img' .. #predictions .. '.jpg' -- bit gross
        print(cmd)
        os.execute(cmd) -- dont think there is cleaner way in Lua
      end
      if verbose then
        print(string.format('image %s: %s predicted: %s',entry.image_id, entry.file_path, entry.caption))
        if opt.split == 'train' then
          print(string.format('image %s: %s gt:       %s', entry.image_id, entry.file_path, entry.gt))
        end
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, num_images)
    if verbose then
      print(string.format('evaluating performance... %d/%d (loss: %f)', 
        ix0-1, ix1, 
        loss))
    end

    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if num_images >= 0 and n >= num_images then break end -- we've used enough images
  end

  local lang_stats
  if not opt.coco_challenge and opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  if n % opt.batch_size * 100 == 0 then collectgarbage() end

  return loss_sum/loss_evals, predictions, lang_stats
end

local loss, split_predictions, lang_stats = 
  eval_split(opt.split, {num_images = opt.num_images, verbose=opt.verbose})
print(string.format('loss: %f', loss))
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
