require 'cutorch'
require 'cudnn'
require 'cunn'
require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
require 'image'

  local cnn_vgg_recons_proto = '/home/jxgu/github/visual_concepts/code/output/vgg/mil_finetune.prototxt.deploy'
  local cnn_vgg_recons_model = '/home/jxgu/github/visual_concepts/code/output/vgg/snapshot_iter_240000.caffemodel'
  cnn = loadcaffe.load(cnn_vgg_recons_proto, cnn_vgg_recons_model, 'cudnn')