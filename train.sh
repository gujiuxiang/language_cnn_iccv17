#!/usr/bin/env bash

clear

case "$1" in
    0)
    echo "Run LSTM +  LCNN model with vgg16 (pre-trained)"
    # cnn GRU
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'LSTM' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_lstm_finetuned.t7'
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'LSTM' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -learning_rate_decay_start 0 -start_from 'checkpoint/bak2/LCNN-LCNN_16IN-RNN-LSTM_best_loss.t7'
    ;;

    1)
    echo "Run RNN +  LCNN model with vgg16 (pre-trained)"
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RNN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rnn_finetuned'
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RNN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -learning_rate_decay_start 0 -start_from 'checkpoint/bak2/LCNN-LCNN_16IN-RNN-RNN_best_loss.t7'
    ;;

    2)
    echo "Run RHN +  LCNN model with vgg16 (pre-trained)"
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10-learning_rate_decay_start 0  -start_cnn_from 'model/bcnn/cnn_512_lcnn_rhn_finetuned.t7'
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -learning_rate_decay_start 0 -start_from 'checkpoint/bak2/LCNN-LCNN_16IN-RNN-RHN_best_loss.t7'
    ;;

    3)
    echo "Run GRU +  LCNN model with vgg16 (pre-trained)"
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'GRU' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10-learning_rate_decay_start 0  -start_cnn_from 'model/bcnn/cnn_512_lcnn_gru_finetuned.t7'
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'GRU' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -learning_rate_decay_start 0 -start_from 'checkpoint/bak2/LCNN-LCNN_16IN-RNN-GRU_best_loss.t7'
    ;;

    4)
    echo "Run LSTM +  LCNN model with vgg16 (pre-trained)"
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -cnn_type 'Resnet101' -rnn_type 'LSTM' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -learning_rate_decay_start 0 -cnn_model 'model/cnn/resnet-101.t7'
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -cnn_type 'Resnet101' -rnn_type 'LSTM' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after 0 -batch_size 10 -learning_rate_decay_start 0 -start_from 'checkpoint/CNN-Resnet101_LCNN-LCNN_16IN-RNN-LSTM_best_score.t7'
    ;;

    5)
    echo "Run RNN +  LCNN model with vgg16 (pre-trained)"
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -cnn_type 'Resnet101' -rnn_type 'RNN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -learning_rate_decay_start 0 -cnn_model 'model/cnn/resnet-101.t7'
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -cnn_type 'Resnet101' -rnn_type 'RNN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after 0 -batch_size 10 -learning_rate_decay_start 0 -start_from 'checkpoint/CNN-Resnet101_LCNN-LCNN_16IN-RNN-RNN_best_score.t7'
    ;;

    6)
    echo "Run RHN +  LCNN model with vgg16 (pre-trained)"
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -cnn_type 'Resnet101' -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -learning_rate_decay_start 0  -cnn_model 'model/cnn/resnet-101.t7'
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -cnn_type 'Resnet101' -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after 0 -batch_size 10 -learning_rate_decay_start 0 -start_from 'checkpoint/CNN-Resnet101_LCNN-LCNN_16IN-RNN-RHN_best_score.t7'
    ;;

    7)
    echo "Run GRU +  LCNN model with vgg16 (pre-trained)"
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -cnn_type 'Resnet101' -rnn_type 'GRU' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -learning_rate_decay_start 0  -cnn_model 'model/cnn/resnet-101.t7'
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -cnn_type 'Resnet101' -rnn_type 'GRU' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after 0 -batch_size 10 -learning_rate_decay_start 0 -start_from 'checkpoint/CNN-Resnet101_LCNN-LCNN_16IN-RNN-GRU_best_score.t7'
    ;;
    #################################################################################### Rebuttel #####################################################
    8)
    echo "Run RHN +  LCNN_16IN model with vgg16 (pre-trained)"
    # cnn RHN
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_from 'checkpoint/CNN-VGG16_LCNN-LCNN_16IN-RNN-RHN_best_score.t7'
    ;;
    
    9)
    echo "Run RHN +  LCNN_16IN_4Layer model with vgg16 (pre-trained)"
    # cnn RHN
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_4Layer' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_4Layer' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_4Layer' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_from 'checkpoint/CNN-VGG16_LCNN-LCNN_16IN_4Layer-RNN-RHN_best_score.t7'
    ;;
    
    10)
    echo "Run RHN +  LCNN_16IN_4Layer_MaxPool model with vgg16 (pre-trained)"
    # cnn RHN
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_4Layer_MaxPool' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_4Layer_MaxPool' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_4Layer_MaxPool' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_from 'checkpoint/CNN-VGG16_LCNN-LCNN_16IN_4Layer_MaxPool-RNN-RHN_best_score.t7'
    ;;
    
    
    11)
    echo "Run RHN +  LCNN_16IN_3Layer model with vgg16 (pre-trained)"
    # cnn RHN
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_3Layer' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_3Layer' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_3Layer' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_from 'checkpoint/CNN-VGG16_LCNN-LCNN_16IN_3Layer-RNN-RHN_best_score.t7'
    ;;
    
    12)
    echo "Run RHN +  LCNN_16IN_3Layer_MaxPool model with vgg16 (pre-trained)"
    # cnn RHN
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_3Layer_MaxPool' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_3Layer' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_3Layer_MaxPool' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_from 'checkpoint/CNN-VGG16_LCNN-LCNN_16IN_3Layer_MaxPool-RNN-RHN_best_score.t7'
    ;;
    
    13)
    echo "Run RHN +  LCNN_16IN_3Layer_MaxPool model with vgg16 (pre-trained)"
    # cnn RHN
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_3Layer_MaxPool' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_8IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -core_width 8 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_8IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -core_width 8 -start_from 'checkpoint/CNN-VGG16_LCNN-LCNN_8IN-RNN-RHN_best_score.t7'
    ;;
    
    14)
    echo "Run RHN +  LCNN_16IN_3Layer_MaxPool model with vgg16 (pre-trained)"
    # cnn RHN
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_3Layer_MaxPool' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    #CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_4IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -core_width 4 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_4IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -core_width 4 -start_from 'checkpoint/CNN-VGG16_LCNN-LCNN_4IN-RNN-RHN_best_score.t7'
    ;;
    
    15)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_3Layer_MaxPool' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7'
    ;;
    
    16)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'NORNN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 1
    #CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'NORNN' -lcnn_type 'LCNN_16IN_NoIm' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_from 'checkpoint/CNN-VGG16_LCNN-LCNN_16IN_NoIm-RNN-NORNN_best_score.t7' -beam_size 2
    ;;
    
    17)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'NORNN_ADD' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 1
    ;;
    
    18)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'LSTM' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 2 -input_h5 'data/coco/cocotalk_karpathy_ol.h5' -input_json 'data/coco/cocotalk_karpathy_ol.json' 
    ;;
    
    
    19)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'NORNN' -lcnn_type 'LCNN_16IN_4Layer_MaxPool' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 1
    ;;
    
    20)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 2 -core_mask 8
    ;;
    
    
    21)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 2 -core_mask 4
    ;;
    
    22)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 2 -core_mask 3
    ;;
    
    23)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 2 -core_mask 6
    ;;
    
    24)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 2 -core_mask 5
    ;;
    
    25)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 2 -core_mask 2
    ;;
    
    26)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=0 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 2 -core_mask 1
    ;;
    
    27)
    echo "Run LCNN_16IN with vgg16 (pre-trained)"
    # cnn RHN
    CUDA_VISIBLE_DEVICES=1 th train.lua -dataset mscoco -limited_gpu 1 -rnn_type 'RHN' -lcnn_type 'LCNN_16IN_33355' -val_images_use 5000 -language_eval 1 -save_checkpoint_every 1000  -finetune_cnn_after -1 -batch_size 10 -start_cnn_from 'model/cnn/cnn_512_lcnn_rhn_finetuned.t7' -beam_size 2 -core_mask 16
    ;;
    
    *)
    echo
    echo "No input"
    ;;
esac
