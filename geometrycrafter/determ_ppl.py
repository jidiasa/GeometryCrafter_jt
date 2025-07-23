from __future__ import annotations
from typing import Optional, Union
import gc
import numpy as np
import jittor as jt
import jittor.nn as nn
from typing import Optional, List, Tuple, Union
import torch
import sys

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipeline,
)

from diffusers.utils import logging
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def create_meshgrid(height, width, normalized=True, dtype="float32"):
    y, x = jt.meshgrid([
        jt.linspace(-1, 1, height) if normalized else jt.arange(height),
        jt.linspace(-1, 1, width) if normalized else jt.arange(width),
    ])
    return jt.stack([x, y], dim=-1)  # [h, w, 2]

@jt.no_grad()
def normalize_point_map(point_map, valid_mask):
    # point_map: [..., 3]
    # valid_mask: [...]
    norm_factor = (point_map[..., 2] * valid_mask.float()).mean() / (valid_mask.float().mean() + 1e-8)
    norm_factor = jt.maximum(norm_factor, 1e-3)
    return point_map / norm_factor

def create_meshgrid(height, width, normalized=True, dtype="float32"):
    y, x = jt.meshgrid([
        jt.linspace(-1, 1, height) if normalized else jt.arange(height),
        jt.linspace(-1, 1, width) if normalized else jt.arange(width),
    ])
    grid = jt.stack([x, y], dim=-1)  # h, w, 2
    return grid

@jt.no_grad()
def point_map_xy2intrinsic_map(point_map_xy):
    # point_map_xy: [..., h, w, 2]
    height, width = point_map_xy.shape[-3], point_map_xy.shape[-2]
    assert height % 2 == 0 and width % 2 == 0

    mesh_grid = create_meshgrid(height, width, normalized=True, dtype=point_map_xy.dtype)
    mesh_grid = mesh_grid.expand(*point_map_xy.shape[:-3], height, width, 2)

    nc = point_map_xy.mean(dim=-3).mean(dim=-2)  # [..., 2]
    nc_map = nc.unsqueeze(-2).unsqueeze(-2).expand_as(point_map_xy)

    nf = ((point_map_xy - nc_map) / mesh_grid).mean(dim=-3).mean(dim=-2)  # [..., 2]
    nf_map = nf.unsqueeze(-2).unsqueeze(-2).expand_as(point_map_xy)

    return jt.concat([nc_map, nf_map], dim=-1)


def robust_min_max(tensor, quantile=0.99):

    T, H, W = tensor.shape
    min_vals = []
    max_vals = []

    for i in range(T):
        flat = tensor[i].reshape(-1)
        flat_sorted = jt.sort(flat)
        n = flat.numel()

        low_idx = int((1 - quantile) * n)
        high_idx = int(quantile * n)
        
        min_vals.append(flat_sorted[low_idx].item())
        max_vals.append(flat_sorted[high_idx].item())

    return min(min_vals), max(max_vals)

nn_functional_interpolate = nn.interpolate  # alias for brevity

def zeros_like(x):
    return jt.zeros_like(x)

def rand_like(x):
    return jt.randn_like(x)

# -----------------------------------------------------------------------------
#  Pipeline
# -----------------------------------------------------------------------------

class GeometryCrafterDetermPipeline(StableVideoDiffusionPipeline):

    @jt.no_grad()
    def encode_video(
        self,
        video: jt.Var,
        chunk_size: int = 14,
    ) -> jt.Var:
        """Encode raw RGB frames to 1024‑d CLIP embeddings.

        video: [T, C, H, W] in **‑1..1**.
        returns: [T, 1024]
        """
        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0

        embs: List[jt.Var] = []
        for i in range(0, video_224.shape[0], chunk_size):
            clip_ready = self.feature_extractor(
                images=video_224[i : i + chunk_size].numpy(),  # extractor still numpy based
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="np",
            ).pixel_values  # np.float32 [B, 3, 224, 224]
            clip_ready = jt.array(clip_ready).astype(video.dtype)
            emb = self.image_encoder(clip_ready).image_embeds  # [B, 1024]
            embs.append(emb)
        return jt.concat(embs, dim=0)

    @jt.no_grad()
    def encode_vae_video(self, video: jt.Var, chunk_size: int = 14) -> jt.Var:
        """Encode frames into VAE latent space.
        video in **‑1..1**. Return [T, C, H/8, W/8]."""
        latents: List[jt.Var] = []
        for i in range(0, video.shape[0], chunk_size):
            lat = self.vae.encode(video[i : i + chunk_size]).latent_dist.mode()
            latents.append(lat)
        return jt.concat(latents, dim=0)

    # ------------------------------------------------------------------
    #  Prior helpers (geometry prediction) – minimal jt rewrite
    # ------------------------------------------------------------------

    @jt.no_grad()
    def produce_priors(self, prior_model, frame: jt.Var, *, chunk_size=8) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var]:
        """Run MoGe prior model on each frame.
        Returns disparity, mask, 3‑D point map, intrinsic map (all Jittor).
        """
        outs: List[Tuple[jt.Var, jt.Var]] = []
        for i in range(0, frame.shape[0], chunk_size):
            p, m = prior_model.forward_image(frame[i : i + chunk_size])
            outs.append((p, m))
        pred_p = jt.concat([o[0] for o in outs], dim=0)
        pred_m = jt.concat([o[1] for o in outs], dim=0)

        pred_m = pred_m.float() * 2 - 1  # mask to ‑1..1
        pred_p = normalize_point_map(pred_p, pred_m > 0)

        # disparity normalisation (same math, jt ops)
        disp = 1.0 / (pred_p[..., 2].clip(min_v=1e-3))
        disp = disp * (pred_m > 0)
        mn, mx = robust_min_max(disp)
        disp = ((disp - mn) / (mx - mn + 1e-4)).clip(0, 1) * 2 - 1

        pred_p[..., :2] = pred_p[..., :2] / (pred_p[..., 2:3] + 1e-7)
        pred_p[..., 2] = (pred_p[..., 2] + 1e-7).log() * (pred_m > 0)

        intr_map = point_map_xy2intrinsic_map(pred_p[..., :2]).permute(0, 3, 1, 2)
        pred_p = pred_p.permute(0, 3, 1, 2)
        return disp, pred_m.squeeze(1), pred_p, intr_map

    # ------------------------------------------------------------------
    #  Point‑map VAE adapters (encode / decode) – JT
    # ------------------------------------------------------------------

    @jt.no_grad()
    def encode_point_map(
        self,
        point_map_vae,
        disparity: jt.Var,
        valid_mask: jt.Var,
        point_map: jt.Var,
        intrinsic_map: jt.Var,
        *,
        chunk_size: int = 8,
    ) -> jt.Var:

        T = point_map.shape[0]
        psuedo_img = disparity.unsqueeze(1).repeat(1, 3, 1, 1)
        intr_norm = jt.norm(intrinsic_map[:, 2:4], p=2, dim=1, keepdims=True)

        lats: List[jt.Var] = []
        for i in range(0, T, chunk_size):
            base_lat = self.vae.encode(psuedo_img[i : i + chunk_size]).latent_dist
            combo = jt.concat([
                intr_norm[i:i+chunk_size],
                point_map[i:i+chunk_size, 2:3],
                disparity[i:i+chunk_size].unsqueeze(1),
                valid_mask[i:i+chunk_size].unsqueeze(1),
            ], dim=1)
            lat_dist = point_map_vae.encode(combo, base_lat)
            lats.append(lat_dist.mode() if hasattr(lat_dist, "mode") else lat_dist)
        return jt.concat(lats, dim=0) * self.vae.config.scaling_factor

    # ------------------------------------------------------------------
    #  Core __call__  (only sketch – denoising loop unchanged)
    # ------------------------------------------------------------------

    @jt.no_grad()
    def __call__(
        self,
        video : Union[np.ndarray, jt.Var],
        point_map_vae,
        prior_model,
        *,
        height               : int = 576,
        width                : int = 1024,
        window_size          : int = 14,
        noise_aug_strength   : float = 0.02,
        decode_chunk_size    : Optional[int] = None,
        overlap              : int = 4,
        force_projection     : bool = True,
        force_fixed_focal    : bool = True,
        use_extract_interp   : bool = False,
        track_time           : bool = False,
        low_memory_usage     : bool = False,                          
    ):
        
        if isinstance(video, np.ndarray):
            video = jt.array(video).permute(0,3,1,2)          # -> [T,C,H,W]
        elif not isinstance(video, jt.Var):
            raise TypeError("`video` must be np.ndarray or jt.Var")

        height = height or video.shape[-2]
        width  = width  or video.shape[-1]
        orig_h, orig_w = int(video.shape[-2]), int(video.shape[-1])
        num_frames     = int(video.shape[0])

        decode_chunk_size = decode_chunk_size or 8
        if num_frames <= window_size:
            window_size, overlap = num_frames, 0
        stride = window_size - overlap

        assert height%64==0 and width%64==0, "`height|width` should be divisible by 64"
        need_resize = (orig_h!=height) or (orig_w!=width)

        device  = self._execution_device    # 依据你的实现
        dtype   = self.dtype
        jt.flags.use_cuda = device.startswith("cuda")


        video_np = video.numpy() if isinstance(video, jt.Var) else video         # [T,C,H,W] -> numpy
        video_pt = torch.as_tensor(video_np, dtype=torch.float32, device=device) # torch tensor

        pred_disp_pt, pred_vmask_pt, pred_pmap_pt, pred_intr_pt = self.produce_priors(
            prior_model,
            video_pt if low_memory_usage else video_pt.to(device),
            chunk_size = decode_chunk_size,
            low_memory_usage = low_memory_usage,
        )

        pred_disp   = jt.array(pred_disp_pt.detach().cpu().numpy())      # [T,H,W]
        pred_vmask  = jt.array(pred_vmask_pt.detach().cpu().numpy())     # [T,H,W]
        pred_pmap   = jt.array(pred_pmap_pt.detach().cpu().numpy())      # [T,3,H,W]
        pred_intr   = jt.array(pred_intr_pt.detach().cpu().numpy())      # [T,2,H,W]

        if need_resize:
            pred_disp  = nn.interpolate(pred_disp.unsqueeze(1), size=(height, width),
                                    mode="bilinear", align_corners=False).squeeze(1)

            pred_vmask = nn.interpolate(pred_vmask.unsqueeze(1), size=(height, width),
                                    mode="bilinear", align_corners=False).squeeze(1)

            z_log      = pred_pmap[:,2:3].clamp_max(10).exp()
            z_log      = nn.interpolate(z_log, size=(height,width), mode="bilinear", align_corners=False).log()
            xy         = nn.interpolate(pred_pmap[:,:2], size=(height,width), mode="bilinear", align_corners=False)
            pred_pmap  = jt.concat([xy, z_log], dim=1)      # [T,3,H,W]

            pred_intr  = nn.interpolate(pred_intr, size=(height,width), mode="bilinear", align_corners=False)
        pred_disp   = jt.array(pred_disp_pt.detach().cpu().numpy())
        pred_vmask  = jt.array(pred_vmask_pt.detach().cpu().numpy())
        pred_pmap   = jt.array(pred_pmap_pt.detach().cpu().numpy())
        pred_intr   = jt.array(pred_intr_pt.detach().cpu().numpy())

        if need_resize:
            video = nn.interpolate(video, size=(height,width), mode='bicubic', align_corners=False, antialias=True)
            video = video.clamp(0,1)

        video = video * 2 - 1          # [0,1]→[-1,1]
        video_embeddings = self.encode_video(video, chunk_size=decode_chunk_size).unsqueeze(0)   # [1,T,1024]

        video_latents = self.encode_vae_video(video.astype(self.vae.dtype), chunk_size=decode_chunk_size)
        video_latents = video_latents.unsqueeze(0).astype(video_embeddings.dtype)      # [1,T,C,H,W]

        prior_latents = self.encode_point_map(
            point_map_vae,
            pred_disp, pred_vmask, pred_pmap, pred_intr,
            chunk_size=decode_chunk_size,
            low_memory_usage=low_memory_usage
        ).unsqueeze(0).astype(video_embeddings.dtype)                                  # [1,T,C,H,W]

        added_time_ids = self._get_add_time_ids(
            7, 127, noise_aug_strength, video_embeddings.dtype,
            batch_size=1, num_videos_per_prompt=1, do_classifier_free_guidance=False
        ).to(device)

        timestep = jt.float32(1.6378)
        self._num_timesteps = 1

        latents_all = None
        idx_start   = 0
        weights     = None
        if overlap>0:
            weights = jt.linspace(0,1, overlap).view(1,overlap,1,1,1).to(device)

        while idx_start < num_frames - overlap:
            idx_end = min(idx_start+window_size, num_frames)

            latents      = prior_latents[:, idx_start:idx_end]
            vid_lat_cur  = video_latents[:, idx_start:idx_end]
            vid_emb_cur  = video_embeddings[:, idx_start:idx_end]

            latent_in = jt.concat([latents, vid_lat_cur], dim=2)

            noise_pred = self.unet(
                latent_in, timestep,
                encoder_hidden_states=vid_emb_cur,
                added_time_ids=added_time_ids,
                return_dict=False
            )[0]

            latents = -1.0 * noise_pred        # c_out = -1

            if latents_all is None:
                latents_all = latents.stop_grad()
            else:
                if overlap>0:
                    latents_all[:, -overlap:] = latents[:, :overlap]*weights + latents_all[:, -overlap:]*(1-weights)
                latents_all = jt.concat([latents_all, latents[:, overlap:]], dim=1)

            idx_start += stride

        latents_all = latents_all.squeeze(0) / self.vae.config.scaling_factor

        point_map, valid_mask = self.decode_point_map(
            point_map_vae, latents_all,
            chunk_size=decode_chunk_size,
            force_projection=force_projection,
            force_fixed_focal=force_fixed_focal,
            use_extract_interp=use_extract_interp,
            need_resize=need_resize,
            height=orig_h, width=orig_w,
            low_memory_usage=low_memory_usage
        )

        self.maybe_free_model_hooks()
        return point_map, valid_mask

