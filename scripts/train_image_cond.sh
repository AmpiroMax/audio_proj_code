#! /bin/bash

dataset_config=~/work/audio_proj/audio_proj_code/configs/dataset_image_cond_config.json
model_config=~/work/audio_proj/audio_proj_code/configs/model_image_cond_config.json
save_dir=~/work/audio_proj/audio_proj_code/data/models
pretrained_ckpt_path=~/work/audio_proj/models/hub/models--stabilityai--stable-audio-open-1.0/snapshots/67351841772475731424cf358c9fb785716ac78f/model.safetensors

conda activate audio_proj
cd ~/work/audio_proj/stable-audio-tools

python3 ~/work/audio_proj/audio_proj_code/src/train/train_image_cond.py \
    --dataset-config ${dataset_config} \
    --model-config ${model_config} \
    --name test_image_cond_train \
    --num-gpus 2 \
    --save-dir ${save_dir} \
    --pretrained-ckpt-path ${pretrained_ckpt_path}