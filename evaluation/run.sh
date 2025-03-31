#!/bin/sh
set -x
set -e

gpu_id=$1
video_path=$2
save_folder=$3
height=$4
width=$5
downsample_ratio=$6
model_type=$7


CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=. python run.py \
  --video_path ${video_path} \
  --save_folder ${save_folder} \
  --cache_dir pretrained_models \
  --height ${height} \
  --width ${width} \
  --downsample_ratio ${downsample_ratio} \
  --model_type ${model_type} \
  --force_fixed_focal False

# CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=. python evaluation/run_moge.py \
#   --video_path ${video_path} \
#   --save_folder ${save_folder} \
#   --cache_dir pretrained_models \
#   --downsample_ratio ${downsample_ratio}