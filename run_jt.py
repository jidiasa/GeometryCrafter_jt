from pathlib import Path
import os, numpy as np, random
from decord import VideoReader, cpu
from fire import Fire

import jittor as jt
from jittor import nn as F                    
jt.flags.use_cuda = 1                         
import sys, pathlib

project_root = pathlib.Path(__file__).parent
extra_dir = project_root / "third_party"
sys.path.insert(0, str(extra_dir)) 

from geometrycrafter import (              
    GeometryCrafterDiffPipeline_jt,
    GeometryCrafterDetermPipeline,
    PMapAutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModelVid2vid,
)

# ---------- util: 统一随机种子 ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)

# ------------------------------------------------------------
def main(
    video_path: str,
    save_folder: str = "workspace/output/",
    cache_dir: str = "workspace/cache",
    height: int = None,
    width: int = None,
    downsample_ratio: float = 1.0,
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 110,
    decode_chunk_size: int = 8,
    overlap: int = 25,
    process_length: int = -1,
    process_stride: int = 1,
    seed: int = 42,
    model_type: str = "diff",        # 'diff' or 'determ'
    force_projection: bool = True,
    force_fixed_focal: bool = True,
    use_extract_interp: bool = False,
    track_time: bool = False,
    low_memory_usage: bool = False,
):
    assert model_type in ("diff", "determ")
    set_seed(seed)

    # ----------- 1. 加载模型 -----------
    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
        "TencentARC/GeometryCrafter",
        subfolder="unet_diff" if model_type == "diff" else "unet_determ",
        cache_dir=cache_dir,
    ).requires_grad_(False)

    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        "TencentARC/GeometryCrafter",
        subfolder="point_map_vae",
        cache_dir=cache_dir,
    ).requires_grad_(False)


    pipe_cls = GeometryCrafterDiffPipeline_jt if model_type == "diff" else GeometryCrafterDetermPipeline
    pipe = pipe_cls.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=unet,
        cache_dir=cache_dir,
    )

    pipe.enable_attention_slicing()

    # ----------- 2. 读视频 -----------
    video_name = os.path.basename(video_path).rsplit(".", 1)[0]
    vr = VideoReader(video_path, ctx=cpu(0))
    ori_h, ori_w = vr.get_batch([0]).shape[1:3]

    height = height or ori_h
    width  = width  or ori_w
    assert height % 64 == 0 and width % 64 == 0

    idx = list(range(0, len(vr), process_stride))
    frames = vr.get_batch(idx).asnumpy().astype("float32") / 255.0  # (T,H,W,C)
    if process_length > 0:
        frames = frames[: min(process_length, len(frames))]
    process_length = len(frames)

    window_size = min(window_size, process_length)
    if window_size == process_length:
        overlap = 0

    # (T,C,H,W) jt.Var
    frames_tensor = jt.Var(frames).permute(0, 3, 1, 2)  # [0,1] float32

    if downsample_ratio > 1.0:
        ori_h, ori_w = frames_tensor.shape[-2:]
        frames_tensor = F.interpolate(
            frames_tensor,
            size=(round(ori_h / downsample_ratio), round(ori_w / downsample_ratio)),
            mode="bicubic",
            align_corners=True,
            antialias=True,
        ).clip(0, 1)
    call_impl = pipe.__call__.__func__
    print(f"__call__ defined in : {call_impl.__module__}.{call_impl.__qualname__}")
    # ----------- 3. 推理 -----------
    with jt.no_grad():
        point_map, valid_mask = pipe(
            frames_tensor,
            point_map_vae,
            prior_model= None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            decode_chunk_size=decode_chunk_size,
            overlap=overlap,
            force_projection=force_projection,
            force_fixed_focal=force_fixed_focal,
            use_extract_interp=use_extract_interp,
            track_time=track_time,
            low_memory_usage=low_memory_usage,
        )

        if downsample_ratio > 1.0:
            point_map = F.interpolate(
                point_map.permute(0, 3, 1, 2),
                size=(ori_h, ori_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            valid_mask = (
                F.interpolate(
                    valid_mask.cast("float32").unsqueeze(1),
                    size=(ori_h, ori_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                > 0.5
            )

        # ----------- 4. 保存 -----------
        out_dir = Path(save_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"{video_name}.npz",
            point_map=point_map.numpy().astype("float16"),
            mask=valid_mask.numpy().astype("bool"),
        )

if __name__ == "__main__":
    Fire(main)
