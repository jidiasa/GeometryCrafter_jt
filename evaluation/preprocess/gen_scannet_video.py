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

DATA_DIR = "/apdcephfs_cq10/share_1290939/stereo_data/open_source_data/OpenDataLab___ScanNet_v2/raw/"
OUTPUT_DIR = f"/path/to/benchmark_datasets/scannet_video/"
SPLIT = "scans_test"
DEPTH_EPS = 1e-5
MAX_SEQ_NUM = 100
MAX_SEQ_LEN = 90*3
FRAME_STRIDE = 3

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

def _read_image(img_rel_path) -> np.ndarray:
    image_to_read = img_rel_path
    image = Image.open(image_to_read)  # [H, W, rgb]
    image = np.asarray(image)
    return image

def depth_read(filename):
    depth_in = _read_image(filename)
    depth_decoded = depth_in / 1000.0
    return depth_decoded

if __name__ == '__main__':
    
    meta_infos = []
    seq_names = sorted(os.listdir(os.path.join(DATA_DIR, SPLIT)))[:MAX_SEQ_NUM]

    for idx, seq_name in enumerate(seq_names):
        rgb_paths = glob.glob(os.path.join(DATA_DIR, SPLIT, seq_name, 'color', '*.jpg'))
        rgb_paths = [os.path.relpath(p, DATA_DIR) for p in rgb_paths]
        rgb_paths = sorted(rgb_paths, key=lambda p: int(os.path.basename(p)[:-4]))

        rgb_paths = rgb_paths[:MAX_SEQ_LEN:FRAME_STRIDE]

        depth_paths = [p.replace('color/', 'depth/').replace('.jpg', '.png') for p in rgb_paths]
        meta_paths = [os.path.join(os.path.dirname(p.replace('color/', 'intrinsic/')), 'intrinsic_depth.txt') for p in rgb_paths]

        os.makedirs(os.path.join(OUTPUT_DIR, SPLIT, seq_name), exist_ok=True)
        
        seq_len = len(rgb_paths)
        progress_bar = tqdm(
            range(seq_len),
        )
        progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

        st_idx = 0
        ed_idx = st_idx + seq_len
        video_save_path = os.path.join(OUTPUT_DIR, SPLIT, seq_name, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
        data_save_path = os.path.join(OUTPUT_DIR, SPLIT, seq_name, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

        meta_infos.append(dict(
            video=os.path.relpath(video_save_path, OUTPUT_DIR),
            data=os.path.relpath(data_save_path, OUTPUT_DIR)
        ))

        frames = []
        disps = []
        point_maps = []
        valid_masks = []

        for rgb_path, depth_path, meta_path in zip(rgb_paths[st_idx:ed_idx], depth_paths[st_idx:ed_idx], meta_paths[st_idx:ed_idx]):
            intrinsic = np.loadtxt(os.path.join(DATA_DIR, meta_path))
            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
            fx, fy, cx, cy=float(fx), float(fy), float(cx), float(cy)
            img = Image.open(os.path.join(DATA_DIR, rgb_path))
            img = np.array(img).astype(np.uint8)
            # grayscale images
            if len(img.shape) == 2:
                img = np.tile(img[..., None], (1, 1, 3))
            else:
                img = img[..., :3]


            depth = depth_read(os.path.join(DATA_DIR, depth_path))
            img = cv2.resize(img, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
            invalid_mask = np.isnan(depth) | (depth < DEPTH_EPS)
            depth[invalid_mask] = DEPTH_EPS
            disp = 1.0 / depth
            disp[invalid_mask] = 0
            valid_mask = np.logical_not(invalid_mask)  

            
            img = img[8:-8, 8:-8, :]
            depth = depth[8:-8, 8:-8]
            disp = disp[8:-8, 8:-8]
            valid_mask = valid_mask[8:-8, 8:-8]
            cx -= 8
            cy -= 8

            if img.shape[0] % 2 != 0  or img.shape[1] % 2 != 0:
                H = img.shape[0] // 2 * 2
                W = img.shape[1] // 2 * 2
                x0, y0 = (img.shape[1] - W) // 2, (img.shape[0] - H) // 2
                img = img[y0:y0+H, x0:x0+W, :]
                depth = depth[y0:y0+H, x0:x0+W]
                disp = disp[y0:y0+H, x0:x0+W]
                valid_mask = valid_mask[y0:y0+H, x0:x0+W]
                cx -= x0
                cy -= y0

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

        imageio.mimsave(video_save_path, frames, fps=24, quality=9, macro_block_size=1)
        with h5py.File(data_save_path, 'w') as h5f:
            h5f.create_dataset('disparity', data=disps.astype(np.float16), chunks=(1, )+disps.shape[1:])
            h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
            h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])
    
    with open(os.path.join(OUTPUT_DIR, 'filename_list.txt'), "w") as f:
        for meta in meta_infos:
            print(meta['video'], meta['data'], file=f)