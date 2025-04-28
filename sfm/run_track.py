import argparse
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import skimage.morphology
from decord import cpu, VideoReader
import sys

from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.models.extractors.sift import SIFT

sys.path.append('sfm/spatracker')

from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer

DEVICE = 'cuda'

def get_query_points(superpoint, sift, query_image, query_mask=None, max_query_num=4096):
    pred_sp = superpoint({"image": query_image})["keypoints"]
    pred_sift = sift({"image": query_image})["keypoints"]

    query_points = torch.cat([pred_sp, pred_sift], dim=1)
    # print(query_points.shape)
    if query_mask is not None:
        assert len(query_image) == 1 and len(query_mask) == 1
        H, W = query_image.shape[-2:]
        grid = query_points.clone()
        grid[:, :, 0] = (grid[:, :, 0] / W) * 2 - 1
        grid[:, :, 1] = (grid[:, :, 1] / H) * 2 - 1 
        grid = grid.unsqueeze(1) # 1, 1, npts, 2
        grid_mask = F.grid_sample(query_mask.unsqueeze(1).float(), grid, mode='bilinear', align_corners=False)
        grid_mask = grid_mask.squeeze() > 0.9
        query_points = query_points[:, grid_mask, :]

    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[:max_query_num]
        query_points = query_points[:, random_point_indices, :]
    return query_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--mask_path", required=True, type=str)
    parser.add_argument("--point_map_path", required=True, type=str)
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--vis_dir', type=str, default='')
    parser.add_argument('--max_query_num_per_frame', type=int, default=1024)
    parser.add_argument('--query_video_length', type=int, default=12)
    parser.add_argument('--query_window_size', type=int, default=6)
    parser.add_argument('--dilation_radius', type=int, default=4)
    parser.add_argument('--spatracker_window_size', type=int, default=12)
    parser.add_argument('--spatracker_checkpoint', type=str, default='pretrained_models/spaT_final.pth')
    parser.add_argument('--superpoint_checkpoint', type=str, default='pretrained_models/superpoint_v6_from_tf.pth')    
    parser.add_argument('--use_ori_res', action='store_true')
    
    args = parser.parse_args()    

    if args.vis_dir != '' and not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    images = []
    masks = []

    vid = VideoReader(args.video_path, ctx=cpu(0))
    images = vid.get_batch(list(range(0, len(vid)))).asnumpy()

    vid = VideoReader(args.mask_path, ctx=cpu(0))
    masks = vid.get_batch(list(range(0, len(vid)))).asnumpy()
    if len(masks.shape) == 4:
        masks = masks[..., 0] > 127
    else:
        masks = masks > 127
    assert images.shape[:-1] == masks.shape
    
    seq_len = images.shape[0]

    # dilate masks & transfer to static masks
    masks = [
        np.logical_not(skimage.morphology.binary_dilation(mask, skimage.morphology.disk(args.dilation_radius)))
        for mask in masks
    ]

    images = torch.from_numpy(images).float().cuda()
    # b, h, w, 3
    masks = torch.from_numpy(np.stack(masks, axis=0)).cuda()
    # b, h, w

    data = np.load(args.point_map_path, allow_pickle=True)
    depths = data['point_map'][..., 2]
    valid_masks = data['mask']
    valid_masks = [~skimage.morphology.binary_dilation(~m, skimage.morphology.disk(args.dilation_radius)) for m in valid_masks]
    depths = torch.from_numpy(depths).float().cuda()
    valid_masks = torch.from_numpy(np.stack(valid_masks, axis=0)).bool().cuda()


    if args.use_ori_res:
        valid_masks = F.interpolate(valid_masks.float().unsqueeze(1), size=images.shape[1:3], mode='nearest-exact').squeeze(1).bool()
        depths = F.interpolate(depths.unsqueeze(1), size=images.shape[1:3], mode='nearest-exact').squeeze(1)
    else:
        images = F.interpolate(images.permute(0, 3, 1, 2), size=depths.shape[1:3], mode='bilinear', antialias=True).permute(0, 2, 3, 1)
        masks = F.interpolate(masks.unsqueeze(1).float(), size=depths.shape[1:3], mode='bilinear').squeeze(1) > 0.5

    superpoint = SuperPoint({
        "nms_radius": 4, 
        "force_num_keypoints": True,
        "max_num_keypoints":args.max_query_num_per_frame,
        "weights": args.superpoint_checkpoint
    }).cuda().eval()
    sift = SIFT({
        "nms_radius": 4, 
        "force_num_keypoints": False,
        "max_num_keypoints":args.max_query_num_per_frame
    }).cuda().eval()

    queries_list = []

    for frame_idx in tqdm(range(0, seq_len, args.query_window_size)):
        if len(images) - frame_idx < args.query_video_length:
            frame_idx = seq_len - args.query_video_length
        frame = images[frame_idx].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        mask = masks[frame_idx].unsqueeze(0) & valid_masks[frame_idx].unsqueeze(0)
        queries = get_query_points(superpoint, sift, frame, mask, max_query_num=args.max_query_num_per_frame)
        queries = queries.squeeze(0)
        # queries_list.append(queries)

        frame_idx2 = min(frame_idx+args.query_video_length, seq_len) - 1
        frame = images[frame_idx2].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        mask = masks[frame_idx2].unsqueeze(0) & valid_masks[frame_idx2].unsqueeze(0)
        queries2 = get_query_points(superpoint, sift, frame, mask, max_query_num=args.max_query_num_per_frame)
        queries2 = queries2.squeeze(0)
        queries_list.append((queries, queries2))

    del sift
    del superpoint

    model = SpaTrackerPredictor(
        checkpoint=args.spatracker_checkpoint,
        interp_shape = (384, 512),
        seq_length = args.spatracker_window_size
    ).eval().cuda()



    for idx, frame_idx in enumerate(tqdm(range(0, seq_len, args.query_window_size))):
        if seq_len - frame_idx < args.query_video_length:
            frame_idx = seq_len - args.query_video_length
        queries, queries2 = queries_list[idx]
        video = images[frame_idx:frame_idx+args.query_video_length].permute(0, 3, 1, 2).unsqueeze(0)
        # 1, t, 3, h, w
        video_depth = depths[frame_idx:frame_idx+args.query_video_length].unsqueeze(1)
        # t, 1, h, w
        query_time = torch.cat([torch.ones_like(queries[:, 0])*0, torch.ones_like(queries2[:, 0])*(video.shape[1]-1)], dim=0)
        queries = torch.cat([
            torch.cat([torch.ones_like(queries[:, :1]) * 0, queries], dim=-1),
            torch.cat([torch.ones_like(queries2[:, :1]) * (video.shape[1]-1), queries2], dim=-1)
        ], dim=0).unsqueeze(0)
        pred_tracks, pred_visibility, T_Firsts = model(
            video, 
            queries=queries,
            video_depth=video_depth,
            wind_length=args.spatracker_window_size
        )
        
        if args.vis_dir:
            vis = Visualizer(
                save_dir=args.vis_dir, grayscale=True, 
                fps=30, pad_value=0, linewidth=1,
                tracks_leave_trace=10
            )
            video_vis = vis.visualize(video=video, tracks=pred_tracks[..., :2],
                                        visibility=pred_visibility,
                                        filename=f"frame{str(frame_idx).zfill(5)}")
        
        pred_tracks = pred_tracks[0][..., :2].detach().cpu().numpy()
        pred_visibility = pred_visibility[0].detach().cpu().numpy()
        query_time = query_time.detach().cpu().numpy()
        np.savez(
            os.path.join(args.out_dir, f'{str(frame_idx).zfill(5)}.npz'), 
            tracks=pred_tracks,
            visible=pred_visibility,
            query_time=query_time)