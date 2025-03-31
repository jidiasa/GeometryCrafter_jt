from pathlib import Path
import torch
from decord import VideoReader, cpu
from fire import Fire
import os
import numpy as np
import torch.nn.functional as F

from third_party import MoGe

def main(
    video_path: str,
    save_folder: str = "workspace/output/",
    cache_dir: str = "pretrained_models", 
    downsample_ratio: float = 1.0,
    chunk_size: int = 16,
    process_length: int = -1,
    process_stride: int = 1,
    force_projection: bool = True,
):
    model = MoGe(
        cache_dir=cache_dir,
    ).requires_grad_(False).to('cuda', dtype=torch.float32)

    video_base_name = os.path.basename(video_path).split('.')[0]
    vid = VideoReader(video_path, ctx=cpu(0))
    original_height, original_width = vid.get_batch([0]).shape[1:3]


    frames_idx = list(range(0, len(vid), process_stride))
    frames = vid.get_batch(frames_idx).asnumpy().astype(np.float32) / 255.0
    if process_length > 0:
        process_length = min(process_length, len(frames))
        frames = frames[:process_length]
    else:
        process_length = len(frames)
    frames_tensor = torch.tensor(frames.astype("float32"), device='cuda').float().permute(0, 3, 1, 2)
    # t,3,h,w

    if downsample_ratio > 1.0:
        original_height, original_width = frames_tensor.shape[-2], frames_tensor.shape[-1]
        frames_tensor = F.interpolate(frames_tensor, (round(frames_tensor.shape[-2]/downsample_ratio), round(frames_tensor.shape[-1]/downsample_ratio)), mode='bicubic', antialias=True).clamp(0, 1)

    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    rec_point_maps = []
    rec_valid_masks = []

    with torch.inference_mode():
        for i in range(0, len(frames_tensor), chunk_size):
            points, masks = model.forward_image(frames_tensor[i:i+chunk_size], force_projection=force_projection)
            rec_point_maps.append(points)
            rec_valid_masks.append(masks)
    rec_point_map = torch.cat(rec_point_maps, dim=0)
    rec_valid_mask = torch.cat(rec_valid_masks, dim=0)


    if downsample_ratio > 1.0:
        rec_point_map = F.interpolate(rec_point_map.permute(0,3,1,2), (original_height, original_width), mode='bilinear').permute(0, 2, 3, 1)
        rec_valid_mask = F.interpolate(rec_valid_mask.float().unsqueeze(1), (original_height, original_width), mode='bilinear').squeeze(1) > 0.5

    np.savez(
        str(save_path / f"{video_base_name}.npz"), 
        point_map=rec_point_map.detach().cpu().numpy().astype(np.float16), 
        mask=rec_valid_mask.detach().cpu().numpy().astype(np.bool_))

if __name__ == "__main__":
    Fire(main)
