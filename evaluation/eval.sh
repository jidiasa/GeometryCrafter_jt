#!/bin/sh
set -x
set -e

methods="GeometryCrafter_diff"
# methods="GeometryCrafter_determ"
# methods="MoGe"
for method in ${methods}; do
    pred_data_root_dir=workspace/benchmark_outputs/${method}
    gt_data_root_dir=workspace/benchmark_datasets

    # eval GMUKitchens
    python evaluation/eval.py \
        --gt_data_dir ${gt_data_root_dir}/GMUKitchens_video \
        --pred_data_dir ${pred_data_root_dir}/GMUKitchens_video \

    # eval scannet
    python evaluation/eval.py \
        --gt_data_dir ${gt_data_root_dir}/scannet_video \
        --pred_data_dir ${pred_data_root_dir}/scannet_video \
        
    # eval DDAD
    python evaluation/eval.py \
        --gt_data_dir ${gt_data_root_dir}/DDAD_video \
        --pred_data_dir ${pred_data_root_dir}/DDAD_video \

    # eval sintel
    python evaluation/eval.py \
        --gt_data_dir ${gt_data_root_dir}/Sintel_video \
        --pred_data_dir ${pred_data_root_dir}/Sintel_video \
        --use_weight

    # eval kitti
    python evaluation/eval.py \
        --gt_data_dir ${gt_data_root_dir}/KITTI_video \
        --pred_data_dir ${pred_data_root_dir}/KITTI_video \
        --use_weight
    
    # eval Monkaa
    python evaluation/eval.py \
        --gt_data_dir ${gt_data_root_dir}/Monkaa_video \
        --pred_data_dir ${pred_data_root_dir}/Monkaa_video \
        --use_weight

done
