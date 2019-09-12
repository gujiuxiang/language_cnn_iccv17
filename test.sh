#!/usr/bin/env bash

clear

case "$1" in
    0)
    echo "Run test"
    CUDA_VISIBLE_DEVICES=0 th  eval_coco_challenge.lua -image_folder '/home/jxgu/github/cvpr2017_im2text_jxgu/torch7/data/coco/test2014/' -split 'test' -coco_json '/home/jxgu/github/cvpr2017_im2text_jxgu/torch7/data/coco/annotations/image_info_test2014.json' -batch_size 25
    ;;
    
    1)
    echo "Run val"
    CUDA_VISIBLE_DEVICES=1 th  eval_coco_challenge.lua -image_folder '/home/jxgu/github/cvpr2017_im2text_jxgu/torch7/data/coco/val2014/' -split 'val' -coco_json '/home/jxgu/github/cvpr2017_im2text_jxgu/torch7/data/coco/annotations/captions_val2014.json' -batch_size 8
    ;;
    
    
    2)
    echo "Run val"
    CUDA_VISIBLE_DEVICES=0 th  test.lua -rnn_type 'NORNN' -lcnn_type 'LCNN_16IN_4Layer_MaxPool' -model 'checkpoint/CNN-VGG16_LCNN-LCNN_16IN_4Layer_MaxPool-RNN-NORNN_best_score.t7' -image_root ''
    ;;
    
    3)
    echo "Run val"
    CUDA_VISIBLE_DEVICES=0 th  test.lua -rnn_type 'NORNN_ADD' -lcnn_type 'LCNN_16IN' -model 'checkpoint/CNN-VGG16_LCNN-LCNN_16IN_4Layer_MaxPool-RNN-NORNN_best_score.t7' -image_root ''
    ;;
    
    4)
    echo "Run val"
    CUDA_VISIBLE_DEVICES=0 th  test.lua -rnn_type 'LSTM' -lcnn_type '' -att_type '' -model 'checkpoint/model_id_basic20160828.t7' -image_root ''
    ;;
    
    *)
    echo
    echo "No input"
    ;;
esac
