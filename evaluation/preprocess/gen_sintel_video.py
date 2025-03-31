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

DATA_DIR = "/path/to/datasets/SintelComplete/"
OUTPUT_DIR = "/path/to/benchmark_datasets/Sintel_video/"
SPLIT = "training"
DEPTH_EPS = 1e-5


# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

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
    
    meta_infos = []
    seq_names = sorted(os.listdir(os.path.join(DATA_DIR, SPLIT, 'clean')))

    for idx, seq_name in enumerate(seq_names):
        rgb_paths = glob.glob(os.path.join(DATA_DIR, SPLIT, 'clean', seq_name, 'frame_*.png'))
        rgb_paths = [os.path.relpath(p, DATA_DIR) for p in rgb_paths]
        rgb_paths = sorted(rgb_paths, key=lambda p: int(os.path.basename(p).split('_')[-1][:-4]))
        depth_paths = [p.replace('clean/', 'depth/').replace('.png', '.dpt') for p in rgb_paths]
        mask_paths = [p.replace('clean/', 'mask/') for p in rgb_paths]
        meta_paths = [p.replace('clean/', 'camdata_left/').replace('.png', '.cam') for p in rgb_paths]
        
        os.makedirs(os.path.join(OUTPUT_DIR, seq_name), exist_ok=True)
        
        seq_len = len(rgb_paths)
        progress_bar = tqdm(
            range(seq_len),
        )
        progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

        st_idx = 0
        ed_idx = seq_len
        video_save_path = os.path.join(OUTPUT_DIR, seq_name, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
        data_save_path = os.path.join(OUTPUT_DIR, seq_name, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

        meta_infos.append(dict(
            video=os.path.relpath(video_save_path, OUTPUT_DIR),
            data=os.path.relpath(data_save_path, OUTPUT_DIR)
        ))

        frames = []
        disps = []
        point_maps = []
        valid_masks = []

        for rgb_path, depth_path, mask_path, meta_path in zip(rgb_paths[st_idx:ed_idx], depth_paths[st_idx:ed_idx], mask_paths[st_idx:ed_idx], meta_paths[st_idx:ed_idx]):
            img = Image.open(os.path.join(DATA_DIR, rgb_path))
            img = np.array(img).astype(np.uint8)
            # grayscale images
            if len(img.shape) == 2:
                img = np.tile(img[..., None], (1, 1, 3))
            else:
                img = img[..., :3]

            depth = depth_read(os.path.join(DATA_DIR, depth_path))
            invalid_mask = Image.open(os.path.join(DATA_DIR, mask_path))
            invalid_mask = np.array(invalid_mask).astype(np.uint8) > 127
            invalid_mask = np.logical_or(depth < 1e-5, invalid_mask)
            depth[invalid_mask] = DEPTH_EPS
            disp = 1.0 / depth
            disp[invalid_mask] = 0.

            valid_mask = np.logical_not(invalid_mask) 

            intr, _ = cam_read(os.path.join(DATA_DIR, meta_path))
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]


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

        frames = center_crop(frames, (436, 872))
        disps = center_crop(disps, (436, 872))
        valid_masks = center_crop(valid_masks, (436, 872))
        point_maps = center_crop(point_maps, (436, 872))

        imageio.mimsave(video_save_path, frames, fps=24, quality=9, macro_block_size=1)
        with h5py.File(data_save_path, 'w') as h5f:
            h5f.create_dataset('disparity', data=disps.astype(np.float16), chunks=(1, )+disps.shape[1:])
            h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
            h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])

        # break
    
    with open(os.path.join(OUTPUT_DIR, 'filename_list.txt'), "w") as f:
        for meta in meta_infos:
            print(meta['video'], meta['data'], file=f)