# An Empirical Study of Language CNN for Image Captioning

This repository contains the code for the following paper:
- Gu, Jiuxiang, Gang Wang, and Tsuhan Chen. ["An Empirical Study of Language CNN for Image Captioning."](https://arxiv.org/pdf/1612.07086.pdf) arXiv preprint arXiv:1612.07086 (2016).
```
@inproceedings{gu2017empirical,
  title={An empirical study of language cnn for image captioning},
  author={Gu, Jiuxiang and Wang, Gang and Cai, Jianfei and Chen, Tsuhan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1222--1231},
  year={2017}
}
```
## Installation
This code is written in Lua and requires Torch. If you're on Ubuntu, installing <b>Torch and a few packages (using LuaRocks)</b> in your home directory may look something like:

```bash
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch;
./install.sh      # and enter "yes" at the end to modify your bashrc
source ~/.bashrc
luarocks install nn
luarocks install nngraph
luarocks install image
luarocks install cutorch
luarocks install cunn
luarocks install torch
luarocks install nn
luarocks install optim
luarocks install lua-cjson
sudo apt-get install libprotobuf-dev protobuf-compiler
luarocks install loadcaffe
sudo apt-get install libhdf5-serial-dev libhdf5-dev
git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec
luarocks install nninit
```

## For training

![](./torch7/model/LanguageCNN_training.png)

1. Firstly, we need to some preprocessing. Head over to the ```COCO/``` folder and run the IPython noetebook to download the dataset and do some preprocessing. The notebook will combine the train/val data together and create a very simple and small json file that contains a large list of image paths, and raw captions for each image, of the form:
```
[{ "file_path": "path/img.jpg", "captions": ["a caption", "a second caption of i"tgit ...] }, ...]
```
2. Once we have this, we're ready to invoke the ```prepro.py``` script, which will read all of this in and create a dataset (an hdf5 file and a json file) ready for consumption in the Lua code. For example, for MS COCO we can run the prepro file as follows:
```bash
python prepro.py --input_json coco/coco_raw.json --num_val 5000 --num_test 5000 --images_root coco/images --word_count_threshold 5 --output_json coco/cocotalk.json --output_h5 coco/cocotalk.h5
# This is telling the script to read in all the data (the images and the captions), allocate 5000 images for val/test splits respectively, and map all words that occur <= 5 times to a special UNK token.
# The resulting json and h5 files are about 30GB and contain everything we want to know about the dataset.
```
3. Put the two pretrained caffe models (the VGG16 prototxt configuration file and the proto binary of weights) somewhere (e.g. a model directory), and we're ready to train!
```bash
th train_cnn_rhw_coco.lua -input_h5 coco/cocotalk.h5 -input_json coco/cocotalk.json
```
The train script will take over, and start dumping checkpoints into the folder specified by checkpoint_path (default = current folder). You also have to point the train script to the VGGNet protos (see the options inside train.lua). If you'd like to evaluate <u>BLEU/METEOR/CIDEr</u> scores during training in addition to validation cross entropy loss, use -language_eval 1 option, but don't forget to download the coco-caption code into coco-caption directory.

## For evaluation
In this case you want to run the evaluation script on a pretrained model checkpoint. I trained a decent one on the [MS COCO dataset](http://mscoco.org/) that you can run on your images. The pretrained checkpoint can be downloaded here: [CNN+LSTM-based COCO](https://pan.baidu.com/s/1pLVmIfT), [CNN+RHW-based COCO](https://pan.baidu.com/s/1pLtZbM7), [CNN+RHW-based Flickr30K](https://pan.baidu.com/s/1miI1Dqw), [CNN+LSTM-based Flickr30K](https://pan.baidu.com/s/1mijpLwc). It's large because it contains the weights of a finetuned VGGNet. Now place all your images of interest into a folder, or run the testing images of MSCOCO, and run the eval script:
```bash
th eval.lua
```
Now visit ```results\*.html``` in your browser and you should see your predicted captions.