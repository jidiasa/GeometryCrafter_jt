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
import cv2

DATA_DIR = "/path/to/datasets/GMUKitchens/"
OUTPUT_DIR = "/path/to/benchmark_datasets/GMUKitchens_video/"
DEPTH_EPS = 1e-5
FRAME_STRIDE = 2
MAX_SEQ_LEN = 110 * FRAME_STRIDE
FOCAL_LENGTH = (1048, 1051)
PRINCIPAL_POINT = (960, 529)


def _read_image(img_rel_path) -> np.ndarray:
    image_to_read = img_rel_path
    image = Image.open(image_to_read)
    image = np.asarray(image)
    return image


def depth_read(filename):
    depth_in = _read_image(filename)
    depth_decoded = depth_in / 1000.0
    return depth_decoded

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
    meta_infos = []

    scenes = sorted(list(filter(lambda x: x.startswith('gmu_scene'), os.listdir(DATA_DIR), )))

    for scene in tqdm(scenes):
        rgb_paths = glob.glob(os.path.join(DATA_DIR, scene, 'Images', 'rgb_*.png'))
        rgb_paths = [os.path.relpath(p, DATA_DIR) for p in rgb_paths]
        rgb_paths = sorted(rgb_paths, key=lambda x: int(os.path.basename(x)[4:-4]))
        
        rgb_paths = rgb_paths[:MAX_SEQ_LEN:FRAME_STRIDE]
        depth_paths = [p.replace('Images', 'Depths').replace('rgb_', 'depth_') for p in rgb_paths]

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

            depth = depth_read(os.path.join(DATA_DIR, depth_path))
            invalid_mask = depth < DEPTH_EPS            
            depth[invalid_mask] = DEPTH_EPS
            disp = 1.0 / depth
            disp[invalid_mask] = 0.
            valid_mask = np.logical_not(invalid_mask)  

            fx, fy = FOCAL_LENGTH
            cx, cy = PRINCIPAL_POINT

            # downsample 
            img = cv2.resize(img, (960, 540),  interpolation=cv2.INTER_CUBIC)
            depth = cv2.resize(depth, (960, 540), interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, (960, 540), interpolation=cv2.INTER_LINEAR)
            valid_mask = cv2.resize(valid_mask.astype(np.float32), (960, 540), interpolation=cv2.INTER_LINEAR) > 0.5
            fx, fy = fx / 2, fy / 2
            cx, cy = cx / 2, cy / 2 

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

        # break
    
    with open(os.path.join(OUTPUT_DIR, 'filename_list.txt'), "w") as f:
        for meta in meta_infos:
            print(meta['video'], meta['data'], file=f)