#!/usr/bin/env bash
# DA
# python3 -u train.py \
#     --pretrained_model_path ./models/google_model.bin \
#     --config_path ./models/google_config.json \
#     --vocab_path ./models/google_vocab.txt \
#     --source bdek.books \
#     --target bdek.dvd \
#     --epochs_num 100 --batch_size 32 \
#     --log_dir log/first_test \
#     --gpus 13 \


# CLIM
CUDA_VISIBLE_DEVICES='' python3 -u train.py \
   --source bdek_backtranslation.kitchen \
   --target bdek_backtranslation.electronics \
   --epochs_num 10 --batch_size 24 \
   --log_dir log/DA_contrastive_KE_1022_01_+indom+MI \
   --num_workers 32 --print_freq 100 --learning_rate 2e-5 \
   --task SSL_DA --model_name SSL_DA_balanced --augmenter back_translation \
   --warmup 0.1 --weight_decay 0.01 --aug_rate 0.7 --initialize uniform \
   --temperature 0.05 --temperature_end 0.05 --temperature_cooldown 0.2 --temperature_scheduler constant \
   --MI_threshold 0.53


# DANN
# CUDA_VISIBLE_DEVICES='1,9,10,11,12,13,14,15' python3 -u train.py \
#       --source bdek.electronics \
#       --target bdek.kitchen \
#       --epochs_num 10 --batch_size 32 \
#       --log_dir log/DANN_EK_1018_02 --num_workers 32 \
#       --task SSL_DA --model_name DANN \
#       --print_freq 100 --augmenter none \
#       --warmup 0.1 --weight_decay 0.01 --gamma 1.0

