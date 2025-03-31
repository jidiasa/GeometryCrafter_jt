#!/bin/sh
set -x
set -e

methods="GeometryCrafter_diff"
# methods="GeometryCrafter_determ"
for method in ${methods}; do
    pred_data_root_dir=workspace/benchmark_outputs/${method}
    gt_data_root_dir=workspace/benchmark_datasets

    # eval scannet
    python evaluation/eval_depth.py \
        --gt_data_dir ${gt_data_root_dir}/scannet_video \
        --pred_data_dir ${pred_data_root_dir}/scannet_video \
        --dataset_max_depth 10

    # eval sintel
    python evaluation/eval_depth.py \
        --gt_data_dir ${gt_data_root_dir}/Sintel_video \
        --pred_data_dir ${pred_data_root_dir}/Sintel_video \
        --dataset_max_depth 70 \
        --use_weight

    # eval kitti
    python evaluation/eval_depth.py \
        --gt_data_dir ${gt_data_root_dir}/KITTI_video \
        --pred_data_dir ${pred_data_root_dir}/KITTI_video \
        --dataset_max_depth 80 \
        --use_weight
        
    # eval DDAD
    python evaluation/eval_depth.py \
        --gt_data_dir ${gt_data_root_dir}/DDAD_video \
        --pred_data_dir ${pred_data_root_dir}/DDAD_video \
        --dataset_max_depth 250 \

    # eval Monkaa
    python evaluation/eval_depth.py \
        --gt_data_dir ${gt_data_root_dir}/Monkaa_video \
        --pred_data_dir ${pred_data_root_dir}/Monkaa_video \
        --dataset_max_depth 100 \
        --use_weight 

    # eval GMUKitchens
    python evaluation/eval_depth.py \
        --gt_data_dir ${gt_data_root_dir}/GMUKitchens_video \
        --pred_data_dir ${pred_data_root_dir}/GMUKitchens_video \
        --dataset_max_depth 10

done
