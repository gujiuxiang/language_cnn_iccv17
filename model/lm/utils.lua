local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.has_value(tab, val)
    for index, value in ipairs(tab) do
        if value == val then
            return true
        end
    end
    return false
end

function utils.read_json(path)
  print(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

function utils.expander_im(batch, images, image_size)
  local repeat_image = torch.CudaTensor(batch*5, 3, image_size, image_size)
  for i=1, batch do
    for j=1, 5 do
      repeat_image[{(i-1)+j}] = images[{i}]:cuda()
    end
  end
  return repeat_image
end

function utils.expander_conv_ft(batch, conv_ft)
  local repeat_conv_ft = torch.CudaTensor(batch*5, 512, 7, 7)
  for i=1, batch do
    for j=1, 5 do
      repeat_conv_ft[{(i-1)+j}] = conv_ft[{i}]:clone()
    end
  end
  return repeat_conv_ft
end

function utils.expander_ft(batch, ft)
  local expand_ft = torch.CudaTensor(batch*5, 196, 512)
  for i=1, batch do
    for j=1, 5 do
      expand_ft[{(i-1)+j}] = ft[{i}]
    end
  end
  return expand_ft
end

-- invert the key and value of a table
function utils.invert_key_value(ix_to_word)
    local word_to_ix = {}
    for ix, word in pairs(ix_to_word) do
        word_to_ix[word] = ix
    end

    return word_to_ix
end 

return utils
