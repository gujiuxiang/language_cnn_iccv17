require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'lfs'
local utils = require 'model.lm.utils'

local text2image = torch.class('text2image')

function text2image:__init(opt)
	self.html 			= utils.getopt(opt, 'html','<html><body><h1>Image Captions and Generated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Image</b></td><td><b>Caption</b></td><td><b>Genearted Images</b></td></tr>')
	self.cur_dir 		= utils.getopt(opt, 'cur_dir', '/home/ajax/gitnas/ntu_lib_utils/lua_utils/library/cls_caption/cls_cvpr2017')
	self.batchSize 		= utils.getopt(opt, 'batchSize', 16) --'number of samples to produce')
	self.noisetype 		= utils.getopt(opt, 'noisetype', 'normal')--'type of noise distribution (uniform / normal')
	self.imsize 		= utils.getopt(opt, 'imsize', 1)--'used to produce larger images. 1:64px. 2:80px, 3:96px, ...')
	self.noisemode 		= utils.getopt(opt, 'noisemode', 'random') --'random / line / linefull1d / linefull')
	self.nz 			= utils.getopt(opt, 'nz', 100)
	self.doc_length 	= utils.getopt(opt, 'doc_length', 201)
	self.queries 		= utils.getopt(opt, 'queries', 'results/coco_queries.txt')
	self.checkpoint_dir = utils.getopt(opt, 'checkpoint_dir', '/home/ajax/gitnas/ntu_lib_utils/lua_utils/library/cls_caption/cls_cvpr2017/checkpoint')
	self.net_gen_path	= utils.getopt(opt, 'net_gen', '20160823coco_nc3_nt256_nz100_bs512_cls_weight0.5_ngf128_ndf64_15_net_G.t7')
	self.net_txt_path	= utils.getopt(opt, 'net_txt', '/home/ajax/gitnas/ntu_lib_utils/lua_utils/library/cls_adversarial/GAN_Text2Image/data/net_txt/coco_gru18_bs64_cls0.5_ngf128_ndf128_a10_c512_80_net_T.t7')
end

function text2image:_loadModel()
	self.net_txt = torch.load(self.net_txt_path)
	self.net_gen = torch.load(self.checkpoint_dir .. '/' .. self.net_gen_path)
	if self.net_txt.protos ~=nil then self.net_txt = self.net_txt.protos.enc_doc end
	self.net_txt:evaluate()
	self.net_gen:evaluate()
	self.net_txt:cuda()
	self.net_gen:cuda()
	
	self.noise = torch.Tensor(self.batchSize, self.nz, self.imsize, self.imsize)
	self.noise = self.noise:cuda()
	if self.noisetype == 'uniform' then self.noise:uniform(-1, 1)
	elseif self.noisetype == 'normal' then  self.noise:normal(0, 1) end
end


-- Extract all text features.
function text2image:extract_text_feature(net_txt, doc_length, queries)
    print('Extracting all txt feature')
    local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    local dict = {}
    for i = 1,#alphabet do dict[alphabet:sub(i,i)] = i end
    ivocab = {}
    for k,v in pairs(dict) do ivocab[v] = k end

    local fea_txt = {}
    -- Decode text for sanity check.
    local raw_txt = {}
    local raw_img = {}
    for query_str in io.lines(queries) do
        print(query_str)
        raw_txt[#raw_txt+1] = query_str
        fea_txt[#fea_txt+1] = text2image:extract_single_text_feature(net_txt, dict, doc_length, query_str)
    end
    return fea_txt, raw_txt
end

-- Extract all text features.
function text2image:extract_single_text_feature(net_txt, dict, doc_length, query)
	local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    local txt = torch.zeros(1,doc_length,#alphabet)
	for t = 1,doc_length do
		local ch = query:sub(t,t)
		local ix = dict[ch]
		if ix ~= 0 and ix ~= nil then
			txt[{1,t,ix}] = 1
		end
	end
	txt = txt:cuda()
	local fea_txt_single = net_txt:forward(txt):clone()
    return fea_txt_single
end

function text2image:generate_image(net_gen, noise, batchSize, fea_txt, raw_txt, dataset, savedir, html)
    print('Generating images')
    --local html = '<html><body><h1>Generated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Caption</b></td><td><b>Image</b></td></tr>'
    for i = 1,#fea_txt do
        print(string.format('generating %d of %d', i, #fea_txt))
        local cur_fea_txt = torch.repeatTensor(fea_txt[i], batchSize, 1)
        local cur_raw_txt = raw_txt[i]
        -------------------------------------------------------------------------------
        local fname_png, fname_txt = text2image:generate_single_image(net_gen, noise, batchSize, cur_fea_txt, dataset, savedir, i)
        html = html .. string.format('\n<tr><td>%s</td><td><img src="%s"></td></tr>',
                                     cur_raw_txt, fname_png)
        os.execute(string.format('echo "%s" > %s', cur_raw_txt, fname_txt))
    end
    
    html = html .. '</html>'
    fname_html = string.format('results/%s.html', dataset)
    os.execute(string.format('echo "%s" > %s', html, fname_html))
end


function text2image:generate_single_image(net_gen, noise, batchSize, cur_fea_txt, dataset, savedir, caption_id)
    -- print('Generating images')
    --local html = '<html><body><h1>Generated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Caption</b></td><td><b>Image</b></td></tr>'
	-------------------------------------------------------------------------------
	local images = net_gen:forward{noise, cur_fea_txt:cuda()}
	local visdir = string.format('%s', dataset)
	--lfs.mkdir('results')
	--lfs.mkdir(visdir)
	local fname = string.format('%s/results/%s/img_%d', savedir, visdir, caption_id)
	local fname_png = fname .. '.png'
	local fname_txt = fname .. '.txt'
	images:add(1):mul(0.5)
	--image.save(fname_png, image.toDisplayTensor(images,4,torch.floor(opt.batchSize/4)))
	image.save(fname_png, image.toDisplayTensor(images,4,batchSize/2))
	return fname_png, fname_txt
end

function text2image:simple_test()
	local fea_txt, raw_txt = text2image:extract_text_feature(self.net_txt, self.doc_length, self.queries)
	text2image:generate_image(self.net_gen, self.noise, self.batchSize, fea_txt, raw_txt, 'coco', self.cur_dir, self.html)
end

