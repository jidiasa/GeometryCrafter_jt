import numpy as np
import h5py
import os
import numpy as np
import torch
import h5py
from tqdm import tqdm
from kornia.geometry.depth import depth_to_3d_v2
import imageio
import sys

sys.path.append('dgp')
from dgp.datasets import SynchronizedSceneDataset

DATA_DIR = "/path/to/datasets/DDAD/ddad_train_val"
OUTPUT_DIR = f"/path/to/benchmark_datasets/DDAD_video/"
DEPTH_EPS = 1e-5

def depth2point_map(depth, fx, fy, cx, cy):
    # [h,w]
    assert len(depth.shape) == 2
    camera_matrix = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ], dtype=depth.dtype, device=depth.device)
    point_maps = depth_to_3d_v2(
        depth,
        camera_matrix,
        normalize_points=False
    ) # h,w,3
    return point_maps



def center_crop(tensor, cropped_size):
    size = tensor.shape[1:3]
    height, width = size
    cropped_height, cropped_width = cropped_size
    needs_vert_crop, top = (
        (True, (height - cropped_height) // 2)
        if height > cropped_height
        else (False, 0)
    )
    needs_horz_crop, left = (
        (True, (width - cropped_width) // 2)
        if width > cropped_width
        else (False, 0)
    )

    needs_crop=needs_vert_crop or needs_horz_crop
    if needs_crop:
        return tensor[:, top:top+cropped_height, left:left+cropped_width]



if __name__ == '__main__':
    
    
    # Load synchronized pairs of camera and lidar frames.
    dataset = SynchronizedSceneDataset(
        os.path.join(DATA_DIR, 'ddad.json'),
        datum_names=('CAMERA_01','lidar'),
        generate_depth_from_datum='lidar',
        split='val'
    )
    # print(len(dataset))
    # print(dataset.dataset_item_index)
    seq_list = dict()
    for meta in dataset.dataset_item_index:
        scene_idx, sample_idx_in_scene, datum_names = meta
        if str(scene_idx) not in seq_list:
            seq_list[str(scene_idx)] = []
        seq_list[str(scene_idx)].append(sample_idx_in_scene)

    # # Iterate through the dataset.
    # for sample in dataset:
    #     camera_01 = sample[0]
    #     image_01 = camera_01['rgb']  # PIL.Image
    #     depth_01 = camera_01['depth'] # (H,W) numpy.ndarray, generated from 'lidar'
        
    meta_infos = []
    for idx, seq_name in enumerate(sorted(seq_list.keys())):
        seq_len = len(seq_list[seq_name])
        os.makedirs(os.path.join(OUTPUT_DIR, 'val', seq_name), exist_ok=True)
        
        progress_bar = tqdm(
            range(seq_len),
        )
        progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_list)})")

        st_idx = 0
        ed_idx = st_idx + seq_len
        video_save_path = os.path.join(OUTPUT_DIR, 'val', seq_name, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
        data_save_path = os.path.join(OUTPUT_DIR, 'val', seq_name, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

        meta_infos.append(dict(
            video=os.path.relpath(video_save_path, OUTPUT_DIR),
            data=os.path.relpath(data_save_path, OUTPUT_DIR)
        ))

        frames = []
        disps = []
        point_maps = []
        valid_masks = []

        for i in range(seq_len):
            scene_idx, sample_idx_in_scene, datum_names = int(seq_name), seq_list[seq_name][i], ['camera_01']
            sample = dataset.get_datum_data(scene_idx, sample_idx_in_scene, 'camera_01')
            # print(sample.keys())
            # print(type(sample['rgb']))
            # print(sample['intrinsics'])
            # print(sample['depth'])
            # continue
            img = sample['rgb']
            img = np.array(img).astype(np.uint8)
            # grayscale images
            if len(img.shape) == 2:
                img = np.tile(img[..., None], (1, 1, 3))
            else:
                img = img[..., :3]

            depth = sample['depth']
            invalid_mask = depth < DEPTH_EPS
            
            depth[invalid_mask] = DEPTH_EPS
            disp = 1.0 / depth
            disp[invalid_mask] = 0.

            valid_mask = np.logical_not(invalid_mask)  

            fx, fy = sample['intrinsics'][0, 0], sample['intrinsics'][1, 1]
            cx, cy = sample['intrinsics'][0, 2], sample['intrinsics'][1, 2]
            
            focal_length = np.array([fx, fy])
            principal_point = np.array([cx, cy])
            point_map = depth2point_map(torch.tensor(depth).float().cuda(), focal_length[0], focal_length[1], principal_point[0], principal_point[1])
            
            frames.append(img)
            disps.append(disp)
            valid_masks.append(valid_mask)
            point_maps.append(point_map.cpu().numpy())

            progress_bar.update(1)
        
        frames = np.stack(frames)
        disps = np.stack(disps)
        valid_masks = np.stack(valid_masks)
        point_maps = np.stack(point_maps)

        frames = center_crop(frames, (1152, 1920))
        disps = center_crop(disps, (1152, 1920))
        valid_masks = center_crop(valid_masks, (1152, 1920))
        point_maps = center_crop(point_maps, (1152, 1920))


        imageio.mimsave(video_save_path, frames, fps=24, quality=9)
        with h5py.File(data_save_path, 'w') as h5f:
            h5f.create_dataset('disparity', data=disps.astype(np.float16), chunks=(1, )+disps.shape[1:])
            h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
            h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])

        # break
    
    with open(os.path.join(OUTPUT_DIR, 'filename_list.txt'), "w") as f:
        for meta in meta_infos:
            print(meta['video'], meta['data'], file=f)