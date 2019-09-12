require 'torch'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'

local LSTM = require 'model.rnn.LSTM'
local GRU = require 'model.rnn.GRU'
local RNN = require 'model.rnn.RNN'
local RHN = require 'model.rnn.RHN'
local LanguageCNN = require 'model.lm.LanguageCNN'

local rnnopt = {}
rnnopt.word_embed_size = 512
rnnopt.input_size = 512
rnnopt.rnn_size = 512
rnnopt.output_size =9568
rnnopt.num_layers = 1

hw = nn.Linear(512, 512)
params_hw, _ = hw:getParameters()
hw1 = nn.Linear(1024, 512)
params_hw1, _ = hw1:getParameters()
print('Highway linear parameters:' .. params_hw:size(1)*3+params_hw1:size(1))

LSTM = LSTM.lstm(rnnopt)
params_lstm, gra1 = LSTM:getParameters()
print ('LSTM parameters:' .. params_lstm:size(1))

RNN = RNN.rnn(rnnopt)
params_rnn, _ = RNN:getParameters()
print ('RNN parameters:' .. params_rnn:size(1))

RHN = RHN.rhn(rnnopt)
params_rhn, _ = RHN:getParameters()
print ('RHN parameters:' .. params_rhn:size(1) )

GRU = GRU.gru(rnnopt)
params_gru, _ = GRU:getParameters()
print ('GRU parameters:' .. params_gru:size(1) )

Linear = nn.Linear(512, 9568)
params_out, _ = Linear:getParameters()
print('Linear parameters:' .. params_out:size(1))

lcnn = LanguageCNN.LanguageCNN_16In() 
params_lcnn, _ = lcnn:getParameters()
print('Language CNN parameters:' .. params_lcnn:size(1))
print('Language CNN + RNN parameters:' .. params_lcnn:size(1)+params_rnn:size(1))