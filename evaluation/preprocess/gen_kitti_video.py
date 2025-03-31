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

DATA_DIR = "/apdcephfs_cq10/share_1290939/stereo_data/open_source_data/KITTI/"
OUTPUT_DIR = f"/path/to/benchmark_datasets/KITTI_video/"
DEPTH_EPS = 1e-5
CLIP_LENGTH = 110
SPLITS = ['val']
FOCAL_LENGTH = None
PRINCIPLE_POINT = None


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


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255

    depth = depth_png.astype(np.float64) / 256.0
    depth[depth_png == 0] = -1.0
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

if __name__ == '__main__':
    
    meta_infos = []

    for split in SPLITS:

        seq_names = sorted(os.listdir(os.path.join(DATA_DIR, 'data_depth_annotated', split)))
        seq_names = list(seq_names)

        for idx, seq_name in enumerate(seq_names):

            depth_paths = glob.glob(os.path.join(DATA_DIR, 'data_depth_annotated', split, seq_name, 'proj_depth', 'groundtruth', 'image_02', '*.png'))
            depth_paths = [os.path.relpath(p, DATA_DIR) for p in depth_paths]
            depth_paths = sorted(depth_paths, key=lambda p: int(os.path.basename(p)[:-4]))
            rgb_paths = [p.replace('data_depth_annotated/val', f'raw_data/{seq_name[:10]}').replace('proj_depth/groundtruth/image_02', 'image_02/data') for p in depth_paths]
            
            PRINCIPLE_POINT = None
            FOCAL_LENGTH = None
            cam_path = os.path.join('raw_data', seq_name[:10], 'calib_cam_to_cam.txt')
            with open(os.path.join(DATA_DIR, cam_path), 'r') as f:
                for line in f.readlines():
                    if line.startswith('P_rect_02'):
                        params = line.split(' ')
                        PRINCIPLE_POINT = float(params[3]), float(params[7])
                        FOCAL_LENGTH = float(params[1]), float(params[6])
                        break


            os.makedirs(os.path.join(OUTPUT_DIR, split, seq_name), exist_ok=True)
            
            seq_len = min(len(rgb_paths), CLIP_LENGTH)
            progress_bar = tqdm(
                range(seq_len),
            )
            progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

            st_idx = 0
            ed_idx = st_idx + seq_len
            video_save_path = os.path.join(OUTPUT_DIR, split, seq_name, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
            data_save_path = os.path.join(OUTPUT_DIR, split, seq_name, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

            meta_infos.append(dict(
                video=os.path.relpath(video_save_path, OUTPUT_DIR),
                data=os.path.relpath(data_save_path, OUTPUT_DIR)
            ))

            frames = []
            disps = []
            point_maps = []
            valid_masks = []

            for rgb_path, depth_path in zip(rgb_paths[st_idx:ed_idx], depth_paths[st_idx:ed_idx]):
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
                cx, cy = PRINCIPLE_POINT

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

            frames = center_crop(frames, (368, 736))
            disps = center_crop(disps, (368, 736))
            valid_masks = center_crop(valid_masks, (368, 736))
            point_maps = center_crop(point_maps, (368, 736))


            imageio.mimsave(video_save_path, frames, fps=24, quality=10, macro_block_size=1)
            with h5py.File(data_save_path, 'w') as h5f:
                h5f.create_dataset('disparity', data=disps.astype(np.float16), chunks=(1, )+disps.shape[1:])
                h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
                h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])

            # break
    
    with open(os.path.join(OUTPUT_DIR, 'filename_list.txt'), "w") as f:
        for meta in meta_infos:
            print(meta['video'], meta['data'], file=f)