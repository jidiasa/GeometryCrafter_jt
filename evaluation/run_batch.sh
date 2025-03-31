set -x
set -e

gpus=0,1,2,3,4,5,6,7
model_type=diff # 'diff' or 'determ'
data_root_dir=workspace/benchmark_datasets
# save_root_dir=workspace/benchmark_outputs/GeometryCrafter_${model_type}
# save_root_dir=workspace/benchmark_outputs/MoGe

python evaluation/run_batch.py \
    --dataset_dir ${data_root_dir}/Monkaa_video/ \
    --save_dir ${save_root_dir}/Monkaa_video/ \
    --height 512 --width 960 \
    --gpus ${gpus} \
    --model_type ${model_type} \

python evaluation/run_batch.py \
    --dataset_dir ${data_root_dir}/Sintel_video/ \
    --save_dir ${save_root_dir}/Sintel_video/ \
    --height 448 --width 896 \
    --gpus ${gpus} \
    --model_type ${model_type} \

python evaluation/run_batch.py \
    --dataset_dir ${data_root_dir}/GMUKitchens_video/ \
    --save_dir ${save_root_dir}/GMUKitchens_video/ \
    --height 512 --width 960 \
    --gpus ${gpus} \
    --model_type ${model_type} \

python evaluation/run_batch.py \
    --dataset_dir ${data_root_dir}/KITTI_video/ \
    --save_dir ${save_root_dir}/KITTI_video/ \
    --height 384 --width 768 \
    --gpus ${gpus} \
    --model_type ${model_type} \


python evaluation/run_batch.py \
    --dataset_dir ${data_root_dir}/scannet_video/ \
    --save_dir ${save_root_dir}/scannet_video/ \
    --height 512 --width 640 \
    --gpus ${gpus} \
    --model_type ${model_type} \


python evaluation/run_batch.py \
    --dataset_dir ${data_root_dir}/DDAD_video/ \
    --save_dir ${save_root_dir}/DDAD_video/ \
    --downsample_ratio 3.0 \
    --height 384 --width 640 \
    --gpus ${gpus} \
    --model_type ${model_type} \