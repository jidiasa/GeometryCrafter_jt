import os
import multiprocessing as mp
import argparse


def process_video(video_path, gpu_id, save_path, args):
    
    os.system(f'sh ./evaluation/run.sh ' + \
              f'{gpu_id} {video_path} {os.path.dirname(save_path)} ' + \
              f'{args.height} {args.width} {args.downsample_ratio} {args.model_type}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--height', type=str, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--downsample_ratio', type=float, default=1.0)
    parser.add_argument('--model_type', type=str, choices=['diff', 'determ'], default='diff')
    parser.add_argument('--gpus', type=str, default="0,1")
    args = parser.parse_args()
    gpus = args.gpus.strip().split(',')

    meta_file_path = os.path.join(args.dataset_dir, 'filename_list.txt')

    samples = []        
    with open(meta_file_path, "r") as f:
        for line in f.readlines():
            video_path, data_path = line.split()
            samples.append(dict(
                video_path=video_path,
                data_path=data_path
            ))
    batch_size = len(gpus)
    sample_batches = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
    print("gpus+++: ", gpus)

    processes = []
    for sample_batch in sample_batches:
        for i, sample in enumerate(sample_batch):
            video_path = os.path.join(args.dataset_dir, sample["video_path"])
            save_path = os.path.join(args.save_dir, sample["video_path"][:-4] + '.npz')
            gpu_id = gpus[i % len(gpus)]
            p = mp.Process(target=process_video, args=(video_path, gpu_id, save_path, args))
            p.start()
            processes.append(p)
        
        for p in processes: 
            p.join()