from typing import Union, Tuple

import jtorch as torch
import sys
sys.path.insert(0, "/home/ubuntu/Desktop/GeometryCrafter/geometrycrafter/stable_video_diffusion/diffusers_jittor/src")

from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.utils import is_torch_version
import jittor as jt
from jittor import nn


class UNetSpatioTemporalConditionModelVid2vid(
    UNetSpatioTemporalConditionModel
):
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def execute(                       # ← Jittor 里 forward ≈ execute
        self,
        sample                : jt.Var,                       # (B,F,C,H,W)
        timestep              : Union[jt.Var, float, int],
        encoder_hidden_states : jt.Var,                       # (B,F,C_hid)
        added_time_ids        : jt.Var,                       # (B,3)
        return_dict           : bool = True,
    ):
        # ---------- 1) 处理 timestep ----------
        if not isinstance(timestep, jt.Var):
            # Jittor 无 MPS；直接按类型选 dtype
            dtype = jt.float32 if isinstance(timestep, float) else jt.int64
            timesteps = jt.Var([timestep], dtype=dtype).to(sample.device)
        else:
            timesteps = timestep
            if timesteps.ndim == 0:
                timesteps = timesteps.reshape(1).to(sample.device)

        B, F = sample.shape[:2]             # batch, frames
        timesteps = timesteps.broadcast((B,))   # (B,)

        # ---------- 2) 时间、增强 Embedding ----------
        t_emb = self.time_proj(timesteps)                # (B,C_t)
        t_emb = t_emb.cast(self.conv_in.weight.dtype)    # 与权重 dtype 对齐
        emb   = self.time_embedding(t_emb)               # (B,C)

        time_embeds = self.add_time_proj(added_time_ids.flatten())  # (B*3,C')
        time_embeds = time_embeds.reshape(B, -1).cast(emb.dtype)
        emb = emb + self.add_embedding(time_embeds)     # (B,C)

        # ---------- 3) reshape 维度 ----------
        sample = sample.reshape(B*F, *sample.shape[2:])           # (B·F,C,H,W)
        emb    = emb.repeat((F,1))                                # (B·F,C)
        encoder_hidden_states = encoder_hidden_states.reshape(B*F, 1, -1) # (B·F,1,C_hid)

        # ---------- 4) pre-process ----------
        sample = sample.cast(self.conv_in.weight.dtype)
        sample = self.conv_in(sample)

        image_only_indicator = jt.zeros((B, F), dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)

        # ---------- 5) 主干网络 ----------
        # if self.training and getattr(self, "gradient_checkpointing", False):

        #     def _ckpt(mod, *args):
        #         # Jittor 的 checkpoint 调用：
        #         #   jt.misc.checkpoint.checkpoint(fn, *inputs)
        #         from jittor.misc.checkpoint import checkpoint
        #         return checkpoint(lambda *i: mod(*i), *args)

        #     for blk in self.down_blocks:
        #         if getattr(blk, "has_cross_attention", False):
        #             sample, res = _ckpt(blk, sample, emb, encoder_hidden_states, image_only_indicator)
        #         else:
        #             sample, res = _ckpt(blk, sample, emb, image_only_indicator)
        #         down_block_res_samples += res

        #     sample = _ckpt(self.mid_block, sample, emb, encoder_hidden_states, image_only_indicator)

        #     for blk in self.up_blocks:
        #         res = down_block_res_samples[-len(blk.resnets):]
        #         down_block_res_samples = down_block_res_samples[:-len(blk.resnets)]
        #         if getattr(blk, "has_cross_attention", False):
        #             sample = _ckpt(blk, sample, res, emb, encoder_hidden_states, image_only_indicator)
        #         else:
        #             sample = _ckpt(blk, sample, res, emb, image_only_indicator)

        for blk in self.down_blocks:
            if getattr(blk, "has_cross_attention", False):
                sample, res = blk(sample, emb, encoder_hidden_states, image_only_indicator)
            else:
                sample, res = blk(sample, emb, image_only_indicator)
            down_block_res_samples += res

        sample = self.mid_block(sample, emb, encoder_hidden_states, image_only_indicator)

        for blk in self.up_blocks:
            res = down_block_res_samples[-len(blk.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(blk.resnets)]
            if getattr(blk, "has_cross_attention", False):
                sample = blk(sample, res, emb, encoder_hidden_states, image_only_indicator)
            else:
                sample = blk(sample, res, emb, image_only_indicator)

        # ---------- 6) post-process ----------
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # ---------- 7) 还原维度 ----------
        sample = sample.reshape(B, F, *sample.shape[1:])     # (B,F,C,H,W)

        if not return_dict:
            return (sample,)
        return dict(sample=sample)
