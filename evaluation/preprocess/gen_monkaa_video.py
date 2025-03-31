import numpy as np
import h5py
import os
from PIL import Image
import numpy as np
import torch
import glob
import h5py
from tqdm import tqdm
from kornia.geometry.depth import depth_to_3d_v2
import imageio
from torchvision.datasets.utils import _read_pfm
import functools

_read_pfm_file = functools.partial(_read_pfm, slice_channels=1)

DATA_DIR = "/path/to/datasets/SceneFlowDataset/Monkaa/"
OUTPUT_DIR = "/path/to/benchmark_datasets/Monkaa_video/"
DEPTH_EPS = 1e-5
DISP_EPS = 1
MAX_SEQ_LEN = 110
FOCAL_LENGTH = 1050
PRINCIPAL_POINT = (479.5, 269.5)


def save_video_disp(filename, video_disp, v_min=None, v_max=None):
    disp = video_disp
    # visualizer = ColorMapper()
    if v_min is None:
        v_min = disp.min()
    if v_max is None:
        v_max = disp.max()
    frames = []
    for i in range(len(disp)):
        disp_img = np.stack([disp[i], disp[i], disp[i]], axis=-1)
        # disp_img = visualizer.apply(torch.tensor(disp[i]), v_min=v_min, v_max=v_max).numpy()
        disp_img = (disp_img * 255).astype(np.uint8)
        frames.append(disp_img)
    frames = np.stack(frames, axis=0)
    imageio.mimsave(filename, frames, fps=24, quality=9) 

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

if __name__ == '__main__':
    
    scene_thres = dict(
        a_rain_of_stones_x2=300,
        funnyworld_x2=100,
        eating_x2=70,
        family_x2=70,
        flower_storm_x2=300,
        lonetree_x2=100,
        top_view_x2=100,
        treeflight_x2=300,
    )

    meta_infos = []

    scenes = sorted(os.listdir(os.path.join(DATA_DIR, 'frames_cleanpass')))

    for scene in tqdm(scenes):
        if scene not in scene_thres:
            continue
        rgb_paths = glob.glob(os.path.join(DATA_DIR, 'frames_cleanpass', scene, 'left', '*.png'))
        rgb_paths = [os.path.relpath(p, DATA_DIR) for p in rgb_paths]
        rgb_paths = sorted(rgb_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        rgb_paths = rgb_paths[:MAX_SEQ_LEN]
        depth_paths = [p.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm') for p in rgb_paths]

        st_idx = 0
        ed_idx = len(rgb_paths)
        os.makedirs(os.path.join(OUTPUT_DIR, scene), exist_ok=True)
        video_save_path = os.path.join(OUTPUT_DIR, scene, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
        data_save_path = os.path.join(OUTPUT_DIR, scene, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

        meta_infos.append(dict(
            video=os.path.relpath(video_save_path, OUTPUT_DIR),
            data=os.path.relpath(data_save_path, OUTPUT_DIR)
        ))
        frames = []
        disps = []
        point_maps = []
        valid_masks = []

        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            img = Image.open(os.path.join(DATA_DIR, rgb_path))
            img = np.array(img).astype(np.uint8)
            # grayscale images
            if len(img.shape) == 2:
                img = np.tile(img[..., None], (1, 1, 3))
            else:
                img = img[..., :3]

            disp = _read_pfm_file(os.path.join(DATA_DIR, depth_path))[0]
            # print(disp.shape, disp.min(), disp.max(), type(disp))
            # assert False
            invalid_mask = disp < DISP_EPS
            disp = np.clip(disp, 1e-3, None)

            fx = fy = FOCAL_LENGTH
            cx, cy = PRINCIPAL_POINT
            depth = 1.0 * FOCAL_LENGTH / disp
            invalid_mask = np.logical_or(invalid_mask, depth > scene_thres[scene])
            depth[invalid_mask] = DEPTH_EPS
            # print(depth.min(), depth.max())
            disp[invalid_mask] = 0.
            disp = disp / img.shape[1]
            valid_mask = np.logical_not(invalid_mask)  

            focal_length = np.array([fx, fy])
            principal_point = np.array([cx, cy])
            point_map = depth2point_map(torch.tensor(depth).float().cuda(), focal_length[0], focal_length[1], principal_point[0], principal_point[1])
            
            frames.append(img)
            disps.append(disp)
            valid_masks.append(valid_mask)
            point_maps.append(point_map.cpu().numpy())

        frames = np.stack(frames)
        disps = np.stack(disps)
        valid_masks = np.stack(valid_masks)
        point_maps = np.stack(point_maps)

        imageio.mimsave(video_save_path, frames, fps=24, quality=10, macro_block_size=1)
        with h5py.File(data_save_path, 'w') as h5f:
            h5f.create_dataset('disparity', data=disps.astype(np.float16), chunks=(1, )+disps.shape[1:])
            h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
            h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])
    
    with open(os.path.join(OUTPUT_DIR, 'filename_list.txt'), "w") as f:
        for meta in meta_infos:
            print(meta['video'], meta['data'], file=f)