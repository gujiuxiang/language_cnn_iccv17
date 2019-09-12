require 'cudnn'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'

local function createDataset(n_in, n_y, seqlen)
  local N, batchSize = 10, 5
  local xs = {}
  local ys = {}
  for i = 1, N do
    xs[i] = torch.rand(seqlen, batchSize, n_in):cuda()
    ys[i] = (torch.rand(seqlen, batchSize) * n_y + 1):int():double():cuda()
  end

  return xs, ys, N
end

local function createModel(opts)
  -- create lstm
  local lstm = cudnn.BLSTM(opts.n_in, opts.n_hid, opts.n_layers)
  if opts.rnn_dropout > 0 then
    lstm.dropout = opts.rnn_dropout
    lstm.seed = opts.seed
    lstm:resetDropoutDescriptor()
  end

  -- create softmax
  local softmax = nn.Sequential()
  if opts.dropout > 0 then
    softmax:add( nn.Dropout(opts.dropout, false, true) )
  end
  softmax:add( nn.Linear(opts.n_hid * 2, opts.n_y) )
  softmax:add( nn.LogSoftMax() )
  softmax = softmax:cuda()

  -- create criterion
  local criterion = nn.ClassNLLCriterion():cuda()

  local m = nn.Parallel()
  m:add(lstm)
  m:add(softmax)
  local params, grads = m:getParameters()

  return {lstm = lstm, softmax = softmax, criterion = criterion,
          params = params, grads = grads}
end

local function main()
  print 'create dataset'
  local opts = {seed = 1, n_in = 50,
                n_hid = 100,
                seqlen = 50,
                n_layers = 2,
                n_y = 10,
                rnn_dropout = 0.2,
                dropout = 0.2,
				gpuid = 0}

  torch.manualSeed(opts.seed)
  cutorch.manualSeed(opts.seed)
  cutorch.setDevice(opts.gpuid + 1)
	
  local xs, ys, size = createDataset(opts.n_in, opts.n_y, opts.seqlen)
  local model = createModel(opts)
  local sgdParam = {learningRate = 0.001}

  for epoch = 1, 10 do
    for i = 1, size do
      local x, y = xs[i], ys[i]
      local function feval(params_)
        if model.params ~= params_ then
          params:copy(params_)
        end
        model.grads:zero()

        -- forward pass
        local hids_ = model.lstm:forward(x)
        local hids = hids_:view(hids_:size(1)*hids_:size(2), hids_:size(3))

        local y_preds = model.softmax:forward(hids)
        local loss = model.criterion:forward(y_preds, y:view(-1))
        print(string.format('epoch = %d, i = %d, loss = %f', epoch, i, loss))

        -- backward
        local df_y_preds = model.criterion:backward(y_preds, y:view(-1))
        local df_hids = model.softmax:backward(hids, df_y_preds)
        local df_hids_ = df_hids:view(hids_:size(1), hids_:size(2), hids_:size(3))
        model.lstm:backward(x, df_hids_)

        return loss, model.grads
      end

      local _, loss_ = optim.adam(feval, model.params, sgdParam)
    end
  end
end

main()