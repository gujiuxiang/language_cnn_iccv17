require 'cutorch'
require 'cunn'
require 'torch'
require 'nn'
require 'optim'
require 'nngraph'
require 'loadcaffe'
-- local imports
require 'model.lm.DataLoader'
require 'model.lm.optim_updates'
local utils = require 'model.lm.utils'
local net_utils = require 'model.lm.net_utils'
require 'model.lm.Attention_Weights_Criterion'
require 'model.lm.L1Criterion'
--------------------------------------------------------------------------------
require 'model.lm.LanguageModel'
--------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
local wordembedding = 'w2v'
local semantic_words = false
local dataset = 'mscoco'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-dataset', 'mscoco','the dataset name ')
cmd:option('-rnn_type','RHN','rnn type')
cmd:option('-lcnn_type', 'LCNN_16IN_4Layer_MaxPool','the cnn model name ')
cmd:option('-core_width', 16,'the cnn model name ')

if semantic_words then
  cmd:option('-semantic_words', true,'use semantic_words')
  cmd:option('-att_type','GATT','rnn type')
  cmd:option('-input_h5', 'data/coco/cocotalk_semantic_words.h5','path to the h5file containing the preprocessed dataset')
  cmd:option('-input_json', 'data/coco/cocotalk_semantic_words.json','path to the json file containing additional info and vocab')
else
  cmd:option('-att_type','','attention type')
  cmd:option('-semantic_words', false,'use semantic_words')
  cmd:option('-input_h5', 'data/coco/cocotalk_0113.h5','path to the h5file containing the preprocessed dataset')
  cmd:option('-input_json', 'data/coco/cocotalk_0113.json','path to the json file containing additional info and vocab')
end
if semantic_words then
  cmd:option('-image_mapping', 1,'add image mapping?')
else
  cmd:option('-image_mapping', 0,'add image mapping?')
end
cmd:option('-core_mask', 16,'the cnn model name ')
cmd:option('-cnn_type', 'VGG16','the cnn model name ')
cmd:option('-cnn_model','model/cnn/resnet-101.t7','path to CNN model file containing the weights.')
cmd:option('-cnn_vgg_proto', 'model/cnn/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_vgg_model', 'model/cnn/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights,')
cmd:option('-cnn_resnet', 'model/cnn/offcial_pretrained/resnet-101.t7')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-start_cnn_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-optim_state_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
-- Model settings
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
if wordembedding == 'w2v' then
  cmd:option('-use_glove', false, 'whether to use glove vector')
  cmd:option('-word_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
  cmd:option('-input_encoding_size', 512,'the encoding size of each token in the vocabulary, and the image.')
  cmd:option('-image_encoding_size', 512,'dimension of the image.')
elseif wordembedding == 'glove' then
  -- glove vector
  cmd:option('-use_glove', true, 'whether to use glove vector')
  cmd:option('-glove_path','glove_word2vec/glove.6B.300d.txt', 'specify glove vector data path')
  cmd:option('-glove_dim',300, 'glove vetor dimension, by default, use 300 dimension')

  cmd:option('-word_encoding_size',300,'the encoding size of each token in the vocabulary, and the image.')
  cmd:option('-input_encoding_size', 300,'the encoding size of each token in the vocabulary, and the image.')
  cmd:option('-image_encoding_size', 300,'dimension of the image.')
end

-- Optimization: General
cmd:option('-max_iters', -1,'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size', 10,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip', 0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5,'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', -1,'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-seq_per_img', 5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- Optimization: for the Language Model
cmd:option('-optim', 'adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate', 1e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha', 0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta', 0.999,'beta used for adam')
cmd:option('-optim_epsilon', 1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the CNN
cmd:option('-cnn_optim', 'adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha', 0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta', 0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate', 1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 5000, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'checkpoint/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-verbose', true, 'true|false')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-limited_gpu', 0, '1|0')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:text()
-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
for k, v in pairs(opt) do print (k, v) end
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
cutorch.manualSeed(opt.seed)
if opt.liminted_gpu then cutorch.setDevice(opt.gpuid + 1) end --lua is 1-indexed

local model_name = 'CNN-' .. opt.cnn_type .. '_LCNN-'.. opt.lcnn_type .. '_RNN-' .. opt.rnn_type .. '' .. opt.core_mask
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json, atts = opt.semantic_words}
local max_train_samples = loader:getTrainSize()
print('Max training samples:', max_train_samples)
-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local protos = {}
local best_score = -1000
local current_score = -1000
local current_loss = -1000
local best_loss = -1000

if string.len(opt.start_from) > 0 then
  print('Initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  -- load past training situation
  --iter = loaded_checkpoint.iter + 1
  --loss_history = loaded_checkpoint.loss_history or loss_history
  --val_lang_stats_history = loaded_checkpoint.val_lang_stats_history or val_lang_stats_history
  --val_loss_history = loaded_checkpoint.val_loss_history or val_loss_history
  --best_score = loaded_checkpoint.best_score
else
  local lmOpt = {}
  lmOpt.image_mapping = opt.image_mapping
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.image_encoding_size = opt.image_encoding_size
  lmOpt.core_width = opt.core_width
  lmOpt.lcnn_type = opt.lcnn_type
  lmOpt.rnn_type = opt.rnn_type
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.att_type = opt.att_type
  lmOpt.attention_size = opt.attention_size
  lmOpt.num_layers = 1
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
  lmOpt.core_mask = opt.core_mask
  -- glove parameters
  lmOpt.use_glove = opt.use_glove
  lmOpt.ix_to_word = loader:getVocab()
  lmOpt.glove_path = opt.glove_path
  lmOpt.glove_dim = opt.glove_dim

  protos.lm = nn.LanguageModel(lmOpt)

  local lm_modules = protos.lm:getModulesList()
  --for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end

  if string.len(opt.start_cnn_from) > 0 then -- initialize the ConvNet
    print('initializing CNN weights from ' .. opt.start_cnn_from)
    protos.cnn = torch.load(opt.start_cnn_from)
  else
    local cnn_raw
    if opt.cnn_type == 'VGG16' then -- load from caffe model
      cnn_raw = loadcaffe.load(opt.cnn_vgg_proto, opt.cnn_vgg_model, cnn_backend)
      protos.cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.image_encoding_size, backend = cnn_backend})
    else
      cnn_raw = torch.load(opt.cnn_model) -- load from pre-saved t7 model
      protos.cnn = net_utils.build_cnn_resnet(cnn_raw, {encoding_size = opt.image_encoding_size, backend = cnn_backend})
    end
  end
  --net_utils.unsanitize_gradients(protos.cnn)
end
--------------------------------------------------------------------------------
-- construct criterion of languge model
--------------------------------------------------------------------------------
protos.crit = nn.LanguageModelCriterion()
--------------------------------------------------------------------------------
protos.expander = nn.FeatExpander(opt.seq_per_img)
-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end
-- flatten and prepare all model parameters to a single vector.
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()

assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())
print('Total number of parameters in LM:' .. params:nElement() .. '; conv CNN:' .. cnn_params:nElement())
-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
-- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias')
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
if opt.image_mapping == 1 then thin_lm.linear_model:share(protos.lm.linear_model, 'weight', 'bias') end
if opt.lcnn_type ~= '' then thin_lm.lcnn:share(protos.lm.lcnn, 'weight', 'bias') end
if opt.semantic_words then
  thin_lm.input_attention_model:share(protos.lm.input_attention_model, 'weight', 'bias')
  if opt.image_mapping == 1 then thin_lm.linear_model:share(protos.lm.linear_model, 'weight', 'bias') end
end
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end
protos.lm:createClones()

local thin_cnn = protos.cnn:clone('weight', 'bias')
net_utils.sanitize_gradients(thin_cnn)

collectgarbage() -- "yeah, sure why not"

logger = optim.Logger('log/model_' .. model_name .. '-' .. os.date('%Y%m') .. 'train.log')
--local dirname = opt.checkpoint_path .. '/' .. opt.model_name
--local checkpoint_path = path.join(dirname, 'model_' .. opt.model_name)
--os.execute("mkdir " .. dirname)
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)
  -- start evaluate
  protos.cnn:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while true do
    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(opt.cnn_type, data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + data.images:size(1)
    -- forward the model to also get generated samples for each image
    local feats
    local exp_attrs
    if opt.semantic_words then -- expand the data.semantic_words: batch_size * 16(attribute words per image)
      exp_attrs = protos.expander:forward(data.semantic_words):clone()
    end
    feats = protos.cnn:forward(data.images)--bs*25088
    local expanded_feats = protos.expander:forward(feats)--bs*5*25088
    local logprobs, attrs, attrs_alpha
    local loss
    if opt.semantic_words then
      logprobs, attrs, attrs_alpha = protos.lm:forward{expanded_feats, data.labels, exp_attrs}
    else
      logprobs = protos.lm:forward{expanded_feats, data.labels}
    end
    loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1
    local seq
    -- forward the model to also get generated samples for each image
    local sample_opts = { sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }

    if opt.semantic_words then
      seq = protos.lm:sample({feats, data.semantic_words})
    else
      seq = protos.lm:sample({feats},sample_opts)
    end

    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end
    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end
    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end
  local lang_stats
  if opt.language_eval == 1 then
    eval_id = protos.lm.lcnn_type .. protos.lm.rnn_type
    lang_stats = net_utils.language_eval(predictions, eval_id, opt.dataset)
  end
  return loss_sum/loss_evals, predictions, lang_stats
end
-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
  protos.cnn:training()
  protos.lm:training()
  grad_params:zero()
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
  data.images = net_utils.prepro(opt.cnn_type, data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
  -- expand the data.semantic_words: batch_size * 16(attribute words per image)
  local exp_attrs
  if opt.semantic_words then
    exp_attrs = protos.expander:forward(data.semantic_words):clone()
  end
  -- data.images: Nx3x224x224
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img
  local feats = protos.cnn:forward(data.images)
  local expanded_feats = protos.expander:forward(feats)
  local dlogprobs, logprobs, attrs, attrs_alpha
  local dexpanded_feats, ddummy
  if opt.semantic_words then
    logprobs = protos.lm:forward{expanded_feats, data.labels, exp_attrs}
  else
    logprobs = protos.lm:forward{expanded_feats, data.labels}
  end
  loss = protos.crit:forward(logprobs, data.labels)
  dlogprobs = protos.crit:backward(logprobs, data.labels)
  if opt.semantic_words then
    dexpanded_feats, ddummy, ddummy = unpack(protos.lm:backward({expanded_feats, data.labels, exp_attrs}, dlogprobs))
  else
    dexpanded_feats, ddummy = unpack(protos.lm:backward({expanded_feats, data.labels}, dlogprobs))
  end

  -- backprop the CNN, but only if we are finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    local dfeats = protos.expander:backward(feats, dexpanded_feats)
    local dx = protos.cnn:backward(data.images, dfeats)
  end
  -- clip gradients
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  -- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -----------------------------------------------------------------------------
  -- and lets get out!
  return {total_loss = loss}
end

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------
local iter_per_epoch = max_train_samples/opt.batch_size
local optim_state = {}
local cnn_optim_state = {}
local valloss = 0
local epoch = 0
local loss0
--------------------------------------------------------------------------------
while true do
  ------------------------------------------------------------------------------
  -- eval loss/gradient
  local losses = lossFun()
  --local optim_state = {learningRate = opt.learning_rate, beta1 = opt.optim_alpha, beta2 = opt.optim_beta, epsilon = opt.optim_epsilon, state = optim_state}
  --local _ , losses = optim.adam(lossFun, params, optim_state)
  ------------------------------------------------------------------------------
  if iter % opt.losses_log_every == 0 then
    loss_history[iter] = losses.total_loss
  end
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('test', {val_images_use = opt.val_images_use, verbose = opt.verbose})
    valloss = val_loss
    print('validation loss: ', val_loss)
    val_loss_history[iter] = val_loss
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history
    write_json_name = opt.checkpoint_path .. 'CNN-' .. opt.cnn_type .. '_LCNN-'.. opt.lcnn_type .. '_RNN-' .. opt.rnn_type .. '.json'
    utils.write_json(write_json_name, checkpoint)
    print('wrote json checkpoint to ' .. write_json_name)

    -- write the full model checkpoint as well if we did better than ever
    ----------------------------------------------------------------------------
    if lang_stats then current_score = lang_stats['CIDEr'] end
    current_loss = -val_loss
    print('current_score : ', current_score )
    ----------------------------------------------------------------------------
    local function save_checkpoint(iter, thin_lm, thin_cnn, loader, save_type)
      if iter > 0 then
        if save_type == 'best_score' then
          best_score = current_score
        else
          best_loss = current_loss
        end
        local save_protos = {} -- include the protos (which have weights) and save to file
        save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
        save_protos.cnn = thin_cnn
        checkpoint.protos = save_protos
        checkpoint.vocab = loader:getVocab()-- also include the vocabulary mapping so that we can use the checkpoint alone to run on arbitrary images without the data loader
        save_checkpoint_name = opt.checkpoint_path .. 'CNN-' .. opt.cnn_type .. '_LCNN-'.. opt.lcnn_type .. '-RNN-' .. opt.rnn_type .. '_' .. save_type .. '.t7'
        torch.save(save_checkpoint_name, checkpoint)
        print('wrote checkpoint to ' .. save_checkpoint_name)
      end
    end
    ----------------------------------------------------------------------------
    if lang_stats then
      if best_score == nil or current_score > best_score then
        save_checkpoint(iter, thin_lm, thin_cnn, loader, 'best_score')
      end
    else
      current_score = 0
    end
    ----------------------------------------------------------------------------
    if best_loss == nil or current_loss > best_loss then
      save_checkpoint(iter, thin_lm, thin_cnn, loader, 'best_loss')
    end
  end

  --xlua.progress(iter, iter_per_epoch)
  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end
  -- perform a parameter update

  if opt.optim == 'rmsprop' then rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else error('bad option for opt.cnn_optim')
    end
  end
  --------------------------------------------------------------------------------
  local finetune_cnn_flag = 0
  if opt.finetune_cnn_after >= 0 then finetune_cnn_flag = 1 end
  epoch = opt.batch_size * iter / max_train_samples
  print(string.format('%s/%2d/%8d, 【Train Loss】lm:%.5f【Val Loss】%.5f/%.5f【Learning Rate】LM:%.5f IM:%.5f　【F/%d】',
      model_name, epoch, iter,
      losses.total_loss,
      valloss, current_score,
      learning_rate, cnn_learning_rate,
      finetune_cnn_flag
  ))
  logger:add{ ['iter'] = iter,
    ['epoch'] = epoch,
    ['batch_size'] = opt.batch_size,
    ['train_loss'] = losses.total_loss,
    ['val_loss'] = valloss,
    ['learning_rate'] = learning_rate,
    ['current_cider_score'] = current_score,
    ['best_cider_score'] = best_score,
    ['best_loss'] = best_loss,
    ['cnn_learning_rate'] = cnn_learning_rate}
  logger:style{['train_loss'] = '-', ['val_loss'] = '-'}
  --------------------------------------------------------------------------------

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
end
