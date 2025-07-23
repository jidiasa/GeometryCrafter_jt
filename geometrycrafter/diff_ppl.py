from typing import Callable, Dict, List, Optional, Union, Tuple
import gc

import numpy as np
import jittor as jt
import jittor.nn as nn
import sys


from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipeline
)
from diffusers.utils import logging
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

import numpy as np, gc, time
from typing import Union, Optional, Callable, Dict, List
from jittor import Var
import inspect

import base64, io, json, subprocess
from PIL import Image
import numpy as np
import jittor as jt

import json, base64, subprocess, os, sys, tempfile, numpy as np, torch

_CHUNK_MAGIC = "__npy__"         
_ENV_NAME    = "geocrafter_torch" 
_WORKER_PY   = os.path.join(os.path.dirname(__file__), "moge_worker.py")

def _start_worker():
    return subprocess.Popen(
        ["conda", "run", "-n", _ENV_NAME, "python", _WORKER_PY],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        bufsize=0  # 行／块全都立即 flush
    )

def _serialize_ndarray(arr: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f:
        np.save(f, arr)
        path = f.name
    return path

def _deserialize_from_path(path: str):
    arr = np.load(path, allow_pickle=False)
    os.remove(path)
    return arr

def call_prior(chunk: torch.Tensor):
    if chunk.device.type != "cpu":
        chunk = chunk.cpu()
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_in, chunk.numpy().astype(np.float32))
    tmp_in.close()

    cmd = [
        "conda", "run", "-n", "geocrafter_torch",
        "python", "moge_worker.py", tmp_in.name
    ]
    out = subprocess.check_output(cmd, text=True)        
    reply = json.loads(out)

    pred_p = np.load(reply["p"]); os.remove(reply["p"])
    pred_m = np.load(reply["m"]); os.remove(reply["m"])
    os.remove(tmp_in.name)                                

    return pred_p, pred_m

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def _to_device(x: Var, device: Optional[str]) -> Var:
    if device is None:
        return x
    target_gpu = str(device).startswith("cuda")
    if target_gpu and not x.is_cuda():
        return x.cuda()
    if not target_gpu and x.is_cuda():
        return x.cpu()
    return x

def retrieve_timesteps(
    scheduler = None,          
    num_inference_steps: Optional[int] = None,
    device = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[Var, int]:
    
    if timesteps is not None:
        if "timesteps" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(
                f"{scheduler.__class__.__name__} 不支持手动指定 timesteps。"
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps_var = scheduler.timesteps
        resolved_steps = len(timesteps_var)

    elif sigmas is not None:
        if "sigmas" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(
                f"{scheduler.__class__.__name__} 不支持手动指定 sigmas。"
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps_var = scheduler.timesteps
        resolved_steps = len(timesteps_var)

    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps_var = scheduler.timesteps
        resolved_steps = num_inference_steps
    if device is not None:
        timesteps_var = _to_device(timesteps_var, device)

    return timesteps_var, resolved_steps

def create_meshgrid(height, width, normalized=True, dtype="float32"):
    y, x = jt.meshgrid([
        jt.linspace(-1, 1, height) if normalized else jt.arange(height),
        jt.linspace(-1, 1, width) if normalized else jt.arange(width),
    ])
    return jt.stack([x, y], dim=-1)  # [h, w, 2]

@jt.no_grad()
def normalize_point_map(point_map, valid_mask):
    norm_factor = (point_map[..., 2] * valid_mask.float()).mean() / (valid_mask.float().mean() + 1e-8)
    norm_factor = jt.maximum(norm_factor, 1e-3)
    return point_map / norm_factor

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

class GeometryCrafterDiffPipeline_jt(StableVideoDiffusionPipeline):

    @jt.no_grad()
    def encode_video(self, video: jt.Var, chunk_size: int = 14):
       
        vid_224 = _resize_with_antialiasing(video.float32(), (224, 224))
        vid_224 = (vid_224 + 1.0) / 2.0  # → [0, 1]
        embeds = []
        for i in range(0, vid_224.shape[0], chunk_size):
            emb = self.feature_extractor(
                images=vid_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",  # Torch tensor expected by CLIP
            ).pixel_values
            # Convert to Jittor and keep dtype
            emb_jt = jt.contrib.torch_converter.torch2jt(emb).to(video.dtype)
            embeds.append(self.image_encoder(emb_jt).image_embeds)
        return jt.concat(embeds, dim=0)

    @jt.no_grad()
    def produce_priors(self, prior_model, frame: jt.Var, *, chunk_size: int = 8, low_memory_usage: bool = False):
        """Run monocular prior network to get disparity/mask/point/intrinsic maps."""
        preds_p, preds_m = [], []
        for i in range(0, len(frame), chunk_size):
            # Prior network is PyTorch‑based → convert
            torch_in = jt.contrib.torch_converter.jt2torch(frame[i : i + chunk_size])
            p_t, m_t = prior_model.forward_image(torch_in)
            preds_p.append(jt.contrib.torch_converter.torch2jt(p_t))
            preds_m.append(jt.contrib.torch_converter.torch2jt(m_t))
        pred_point = jt.concat(preds_p, dim=0)
        pred_mask = jt.concat(preds_m, dim=0)

        pred_mask = pred_mask.float32() * 2 - 1
        pred_point = self._normalize_point_map(pred_point, pred_mask > 0)

        pred_disp = 1.0 / jt.maximum(pred_point[..., 2], 1e-3)
        pred_disp = pred_disp * (pred_mask > 0)
        min_d, max_d = robust_min_max(pred_disp)
        pred_disp = (pred_disp - min_d) / (max_d - min_d + 1e-4)
        pred_disp = jt.clamp(pred_disp, 0, 1) * 2 - 1

        pred_point[..., :2] = pred_point[..., :2] / (pred_point[..., 2:3] + 1e-7)
        pred_point[..., 2] = jt.log(pred_point[..., 2] + 1e-7) * (pred_mask > 0)
        pred_intr = self._point_map_xy2intrinsic(pred_point[..., :2]).permute(0, 3, 1, 2)
        pred_point = pred_point.permute(0, 3, 1, 2)
        return pred_disp, pred_mask, pred_point, pred_intr
    
    @jt.no_grad()
    def encode_vae_video(self, video: jt.Var, chunk_size: int = 14):
        """VAE encode video to latent space (T, C, H, W)."""
        lat = []
        for i in range(0, video.shape[0], chunk_size):
            frame = video[i : i + chunk_size]
            # Convert to Torch for diffusers' VAE
            lat_torch = self.vae.encode(jt.contrib.torch_converter.jt2torch(frame)).latent_dist.mode()
            lat.append(jt.contrib.torch_converter.torch2jt(lat_torch))
        return jt.concat(lat, dim=0)
    
    @jt.no_grad()
    def produce_priors(self, prior_model, frame: jt.Var, *, chunk_size: int = 8, low_memory_usage: bool = False):
        """Run monocular prior network to get disparity/mask/point/intrinsic maps."""
        T, _, H, W = frame.shape
        pred_point_maps, pred_masks = [], []

        if not hasattr(self, "_torch_worker"):
            self._torch_worker = _start_worker()

        worker = self._torch_worker

        for i in range(0, T, chunk_size):
            chunk = frame[i:i+chunk_size]
            if low_memory_usage:
                chunk = chunk.to(self._execution_device)      # e.g. CPU
            pred_p, pred_m = call_prior(chunk)

            pred_point_maps.append(pred_p)
            pred_masks.append(pred_m)

        if hasattr(self, "_torch_worker"):
            self._torch_worker.stdin.close()
            self._torch_worker.terminate()
            del self._torch_worker

        pred_point_maps = jt.array(np.concatenate(pred_point_maps, axis=0))
        pred_masks      = jt.array(np.concatenate(pred_masks,      axis=0))

        pred_masks = pred_masks.cast(jt.float32) * 2 - 1

        pred_point_maps = normalize_point_map(pred_point_maps, pred_masks > 0)

        # disparity = 1/z
        depth          = jt.maximum(pred_point_maps[..., 2], 1e-3)
        pred_disps     = 1.0 / depth
        pred_disps    *= (pred_masks > 0)

        min_disp, max_disp = robust_min_max(pred_disps)
        pred_disps = (pred_disps - min_disp) / (max_disp - min_disp + 1e-4)
        pred_disps = jt.clamp(pred_disps, 0.0, 1.0) * 2 - 1 

        pred_point_maps[..., :2] = pred_point_maps[..., :2] / (pred_point_maps[..., 2:3] + 1e-7)
        pred_point_maps[...,  2] = jt.log(pred_point_maps[..., 2] + 1e-7) * (pred_masks > 0)
        pred_intr_maps = point_map_xy2intrinsic_map(pred_point_maps[..., :2]).permute(0, 3, 1, 2)
        pred_point_maps = pred_point_maps.permute(0, 3, 1, 2)

        return pred_disps, pred_masks, pred_point_maps, pred_intr_maps
    
    @jt.no_grad()         
    def encode_point_map(self,
                            point_map_vae,           
                            disparity,                # (T,H,W)
                            valid_mask,               # (T,H,W)
                            point_map,                # (T,3,H,W)
                            intrinsic_map,            # (T,4,H,W)——和原逻辑保持一致
                            chunk_size: int = 8,
                            low_memory_usage: bool = False):
        T, _, H, W = point_map.shape
        latents = []

        psedo_image = disparity[:, None].repeat(1, 3, 1, 1)

        intrinsic_map = jt.norm(intrinsic_map[:, 2:4], p=2, dim=1, keepdim=False)

        for i in range(0, T, chunk_size):
            if low_memory_usage:
                chunk_img = chunk_img.to(self._execution_device)
            chunk_img = chunk_img.to(self.vae.dtype)
            latent_dist = self.vae.encode(chunk_img).latent_dist  
            concat_in = jt.concat([
                intrinsic_map[i : i + chunk_size, None].to(self._execution_device) if low_memory_usage else intrinsic_map[i : i + chunk_size, None],
                point_map   [i : i + chunk_size, 2:3].to(self._execution_device)    if low_memory_usage else point_map   [i : i + chunk_size, 2:3],
                disparity   [i : i + chunk_size, None].to(self._execution_device)   if low_memory_usage else disparity   [i : i + chunk_size, None],
                valid_mask  [i : i + chunk_size, None].to(self._execution_device)   if low_memory_usage else valid_mask  [i : i + chunk_size, None],
            ], dim=1)

            latent_dist = point_map_vae.encode(concat_in, latent_dist)
            if DiagonalGaussianDistribution is not None and isinstance(latent_dist, DiagonalGaussianDistribution):
                latent = latent_dist.mode()
            else:
                latent = latent_dist

            assert isinstance(latent, jt.Var)
            latents.append(latent)
        latents = jt.concat(latents, dim=0)
        latents = latents * self.vae.config.scaling_factor
        return latents

    @jt.no_grad()                   
    def decode_point_map(
        self,                       
        point_map_vae, 
        latents, 
        chunk_size: int      = 8,
        force_projection: bool = True,
        force_fixed_focal: bool = True,
        use_extract_interp: bool = False,
        need_resize: bool   = False,
        height = None,
        width = None,
        low_memory_usage: bool = False
    ):
        T = latents.shape[0]
        rec_intrinsic_maps, rec_depth_maps, rec_valid_masks = [], [], []

        for i in range(0, T, chunk_size):
            lat = latents[i : i + chunk_size]
            rec_imap, rec_dmap, rec_vmask = point_map_vae.decode(
                lat,
                num_frames = lat.shape[0],
            )
            if low_memory_usage:          
                rec_imap  = rec_imap.to('cpu')
                rec_dmap  = rec_dmap.to('cpu')
                rec_vmask = rec_vmask.to('cpu')

            rec_intrinsic_maps.append(rec_imap)
            rec_depth_maps    .append(rec_dmap)
            rec_valid_masks   .append(rec_vmask)

        rec_intrinsic_maps = jt.concat(rec_intrinsic_maps, dim=0)
        rec_depth_maps     = jt.concat(rec_depth_maps,     dim=0)
        rec_valid_masks    = jt.concat(rec_valid_masks,    dim=0)

        if need_resize:
            size = (height, width)
            if use_extract_interp:                             
                rec_depth_maps  = nn.interpolate(rec_depth_maps,  size=size, mode='nearest')
                rec_valid_masks = nn.interpolate(rec_valid_masks, size=size, mode='nearest')
            else:
                rec_depth_maps  = jt.log(
                    nn.interpolate(
                        jt.exp(jt.minimum(rec_depth_maps, 10.0)),
                        size=size, mode='bilinear', align_corners=False
                    )
                )
                rec_valid_masks = nn.interpolate(rec_valid_masks, size=size, mode='bilinear', align_corners=False)

            rec_intrinsic_maps = nn.interpolate(
                rec_intrinsic_maps, size=size, mode='bilinear', align_corners=False
            )
        H, W = rec_intrinsic_maps.shape[-2], rec_intrinsic_maps.shape[-1]

        grid_y, grid_x = jt.meshgrid(jt.linspace(-1, 1, H), jt.linspace(-1, 1, W))
        mesh_grid = jt.stack([grid_x, grid_y], dim=-1)[None]        
        mesh_grid = mesh_grid.permute(0, 3, 1, 2).to(               
            rec_intrinsic_maps.device, rec_intrinsic_maps.dtype
        )

        factor = jt.sqrt(W * W + H * H)
        rec_intrinsic_maps = jt.concat(
            [
                rec_intrinsic_maps * W / factor,    # fx'
                rec_intrinsic_maps * H / factor     # fy'
            ], dim=1
        )   # (T,2,H,W)

        rec_valid_masks = (rec_valid_masks.squeeze(1) > 0)          
        if force_projection:
            vm_float = rec_valid_masks.cast(jt.float32)
            if force_fixed_focal:
                nfx = (rec_intrinsic_maps[:, 0] * vm_float).mean() / (vm_float.mean() + 1e-4)
                nfy = (rec_intrinsic_maps[:, 1] * vm_float).mean() / (vm_float.mean() + 1e-4)
                rec_intrinsic_maps = jt.stack([nfx, nfy])[None, :, None, None].repeat(T, 1, 1, 1)
            else:
                nfx = (rec_intrinsic_maps[:, 0] * vm_float).mean(dim=[-2, -1]) / (vm_float.mean(dim=[-2, -1]) + 1e-4)
                nfy = (rec_intrinsic_maps[:, 1] * vm_float).mean(dim=[-2, -1]) / (vm_float.mean(dim=[-2, -1]) + 1e-4)
                rec_intrinsic_maps = jt.stack([nfx, nfy], dim=-1)[:, :, None, None]    # (T,2,1,1)

        rec_point_maps = jt.concat(
            [rec_intrinsic_maps * mesh_grid, rec_depth_maps], dim=1
        ).permute(0, 2, 3, 1)     
        xy, z = jt.split(rec_point_maps, [2, 1], dim=-1)
        z = jt.exp(jt.minimum(z, 10.0))         
        rec_point_maps = jt.concat([xy * z, z], dim=-1)   

        return rec_point_maps, rec_valid_masks      

    @jt.no_grad()                                    # ↔︎ @torch.no_grad()
    def __call__(
        self,
        video           : Union[np.ndarray, jt.Var], # (T,H,W,C) or (T,C,H,W), range [0,1]
        point_map_vae,
        prior_model,
        *,
        height          : int = 576,
        width           : int = 1024,
        num_inference_steps : int = 5,
        guidance_scale  : float = 1.0,
        window_size     : Optional[int] = 14,
        noise_aug_strength : float = 0.02,
        decode_chunk_size  : Optional[int] = None,
        generator = None,
        latents         : Optional[jt.Var] = None,
        callback_on_step_end: Optional[Callable[[int,int,Dict],None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        overlap         : int  = 4,
        force_projection: bool = True,
        force_fixed_focal: bool = True,
        use_extract_interp: bool = False,
        track_time      : bool = False,
        low_memory_usage: bool = False
    ):
        # ---------- 0. 输入规范化 ----------
        if isinstance(video, np.ndarray):
            video = jt.Var(video.transpose(0, 3, 1, 2))     # (T,C,H,W)
        else:
            assert isinstance(video, jt.Var)

        height  = height  or video.shape[-2]
        width   = width   or video.shape[-1]
        oH, oW  = video.shape[-2], video.shape[-1]
        T       = video.shape[0]
        decode_chunk_size = decode_chunk_size or 8

        if T <= window_size:
            window_size, overlap = T, 0
        stride = window_size - overlap

        assert height % 64 == 0 and width % 64 == 0
        need_resize = (oH != height) or (oW != width)


        self._guidance_scale = guidance_scale
        if track_time: t0 = time.perf_counter()

        pred_disp, pred_mask, pred_pmap, pred_imap = self.produce_priors(
            prior_model,
            video.cast(jt.float32) if low_memory_usage else video.cast(jt.float32).to(self._execution_device),
            chunk_size = decode_chunk_size,
            low_memory_usage = low_memory_usage,
        )                                                  # 形状与原版一致

        # optional resize
        if need_resize:
            pred_disp  = nn.interpolate(pred_disp.unsqueeze(1),  size=(height,width), mode='bilinear', align_corners=False).squeeze(1)
            pred_mask  = nn.interpolate(pred_mask.unsqueeze(1),  size=(height,width), mode='bilinear', align_corners=False).squeeze(1)
            pred_pmap  = jt.concat([
                nn.interpolate(pred_pmap[:,0:2],             size=(height,width), mode='bilinear', align_corners=False),
                nn.interpolate(jt.exp(jt.minimum(pred_pmap[:,2:3],10.0)), size=(height,width), mode='bilinear', align_corners=False).log()
            ], dim=1)
            pred_imap  = nn.interpolate(pred_imap, size=(height,width), mode='bilinear', align_corners=False)

        if track_time:
            print(f'[timer] prior  : {(time.perf_counter()-t0)*1e3:.1f} ms'); t1 = time.perf_counter()
        else:
            gc.collect(); jt.sync_all()

        if need_resize:
            video = nn.interpolate(video, size=(height,width), mode="bicubic", align_corners=False).clamp(0,1)
        video = video.cast(self.dtype) * 2.0 - 1.0                          # [-1,1]
        video_emb = self.encode_video(video, chunk_size=decode_chunk_size).unsqueeze(0)

        needs_upcast = (self.vae.dtype == jt.float16 and self.vae.config.force_upcast)
        if needs_upcast: self.vae.to(dtype=jt.float32)

        video_lat   = self.encode_vae_video(video.cast(self.vae.dtype),
                                            chunk_size=decode_chunk_size).unsqueeze(0).cast(video_emb.dtype)

        if low_memory_usage: del video; gc.collect(); jt.sync_all()

        prior_lat   = self.encode_point_map_jt(
            point_map_vae,
            pred_disp, pred_mask, pred_pmap, pred_imap,
            chunk_size=decode_chunk_size,
            low_memory_usage=low_memory_usage
        ).unsqueeze(0).cast(video_emb.dtype)

        if track_time:
            print(f'[timer] encode : {(time.perf_counter()-t1)*1e3:.1f} ms'); t2 = time.perf_counter()
        else:
            gc.collect(); jt.sync_all()

        if needs_upcast: self.vae.to(dtype=jt.float16)

        added_time_ids = self._get_add_time_ids(
            7, 127, noise_aug_strength, video_emb.dtype, 1, 1, False
        ).to(self._execution_device)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self._execution_device, None, None
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        num_channels_latents = 8  # = in_channels -  prior_channels
        latents_init = self.prepare_latents(
            1, window_size, num_channels_latents, height, width,
            video_emb.dtype, self._execution_device, generator, latents
        )                                  # (1,T,C,H,W)
        latents_all = None

        if overlap > 0:
            weights = jt.linspace(0,1,overlap).view(1,overlap,1,1,1).to(self._execution_device)
        else:
            weights = None

        idx_start = 0
        while idx_start < T - overlap:
            idx_end = min(idx_start + window_size, T)
            self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)

            latents      = latents_init[:, :idx_end-idx_start].clone()
            latents_init = jt.concat([latents_init[:, -overlap:], latents_init[:, :stride]], dim=1)

            v_lat_cur  = video_lat [:, idx_start:idx_end]
            p_lat_cur  = prior_lat [:, idx_start:idx_end]
            v_emb_cur  = video_emb [:, idx_start:idx_end]
            with self.progress_bar(total=num_inference_steps) as pbar:
                for i, t in enumerate(timesteps):
                    if latents_all is not None and i == 0 and overlap>0:
                        latents[:, :overlap] = (
                            latents_all[:, -overlap:]
                            + latents[:, :overlap] / self.scheduler.init_noise_sigma
                            * self.scheduler.sigmas[i]
                        )

                    l_in = self.scheduler.scale_model_input(latents, t)
                    l_in = jt.concat([l_in, v_lat_cur, p_lat_cur], dim=2)

                    noise_pred = self.unet(
                        l_in, t,
                        encoder_hidden_states=v_emb_cur,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]

                    # classifier-free guidance
                    if self.do_classifier_free_guidance:
                        l_in_uc = self.scheduler.scale_model_input(latents, t)
                        l_in_uc = jt.concat([l_in_uc,
                                            jt.zeros_like(l_in_uc),
                                            jt.zeros_like(l_in_uc)], dim=2)
                        noise_uncond = self.unet(
                            l_in_uc, t,
                            encoder_hidden_states=jt.zeros_like(v_emb_cur),
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + self.guidance_scale * (noise_pred - noise_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                    if callback_on_step_end is not None:
                        cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                        latents = callback_on_step_end(self, i, t, cb_kwargs).get("latents", latents)

                    if i == len(timesteps)-1 or ((i+1) > num_warmup_steps and (i+1)%self.scheduler.order==0):
                        pbar.update()

            # 拼接 window 结果
            if latents_all is None:
                latents_all = latents.clone()
            else:
                if overlap>0:
                    latents_all[:, -overlap:] = latents[:, :overlap]*weights + latents_all[:, -overlap:]*(1-weights)
                latents_all = jt.concat([latents_all, latents[:, overlap:]], dim=1)

            idx_start += stride

        latents_all = latents_all.squeeze(0) / self.vae.config.scaling_factor
        latents_all = latents_all.cast(jt.float32)

        if low_memory_usage:
            del latents, prior_lat, video_lat; gc.collect(); jt.sync_all()

        if track_time:
            print(f'[timer] denoise: {(time.perf_counter()-t2)*1e3:.1f} ms'); t3 = time.perf_counter()
        else:
            gc.collect(); jt.sync_all()

        point_map, valid_mask = self.decode_point_map_jt(
            point_map_vae, latents_all,
            chunk_size = decode_chunk_size,
            force_projection = force_projection,
            force_fixed_focal = force_fixed_focal,
            use_extract_interp = use_extract_interp,
            need_resize = need_resize,
            height = oH, width = oW,
            low_memory_usage = low_memory_usage
        )

        if track_time:
            print(f'[timer] decode : {(time.perf_counter()-t3)*1e3:.1f} ms')
        else:
            gc.collect(); jt.sync_all()

        self.maybe_free_model_hooks()
        return point_map, valid_mask        
