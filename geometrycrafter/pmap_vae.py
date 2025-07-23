from typing import Dict, Tuple, Union

import sys

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, Encoder
from diffusers.utils import is_torch_version
from diffusers.models.unet_3d_blocks import UpBlockTemporalDecoder, MidBlockTemporalDecoder
from diffusers.models.resnet import SpatioTemporalResBlock

import jittor as jt
from jittor import nn
from typing import Tuple, List

def zero_module(module: nn.Module):

    for p in module.parameters():

        p.stop_grad()       
        p.assign(0)

    return module

class PMapTemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: Tuple[int] = (1, 1, 1),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()

        # -------- 1. 首层卷积 --------
        self.conv_in = nn.Conv(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1
        )

        # -------- 2. 中间块 --------
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
        )

        # -------- 3. 上采样链 --------
        self.up_blocks = nn.ModuleList()
        rev_channels: List[int] = list(reversed(block_out_channels))
        out_ch = rev_channels[0]
        for i in range(len(block_out_channels)):
            prev_out = out_ch
            out_ch   = rev_channels[i]
            self.up_blocks.append(
                UpBlockTemporalDecoder(
                    num_layers      = layers_per_block + 1,
                    in_channels     = prev_out,
                    out_channels    = out_ch,
                    add_upsample    = i != len(block_out_channels) - 1,
                )
            )

        # -------- 4. 输出分支 --------
        self.out_blocks = nn.ModuleList()
        self.time_conv_outs = nn.ModuleList()
        for oc in out_channels:
            self.out_blocks.append(
                nn.ModuleList([
                    nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6),
                    nn.ReLU(),
                    nn.Conv(block_out_channels[0], block_out_channels[0]//2, kernel_size=3, padding=1),
                    SpatioTemporalResBlock(
                        in_channels  = block_out_channels[0]//2,
                        out_channels = block_out_channels[0]//2,
                        temb_channels=None,
                        eps=1e-6,
                        temporal_eps=1e-5,
                        merge_factor=0.0,
                        merge_strategy="learned",
                        switch_spatial_to_temporal_mix=True,
                    ),
                    nn.ReLU(),
                    nn.Conv(block_out_channels[0]//2, oc, kernel_size=1),
                ])
            )

            k3d = (3, 1, 1)
            pad = tuple(int(k//2) for k in k3d)
            self.time_conv_outs.append(
                nn.Conv3d(oc, oc, kernel_size=k3d, padding=pad)
            )

    def execute(
        self,
        sample: jt.Var,                 # (B·F,C,H,W)
        image_only_indicator: jt.Var,   # (B,F)
        num_frames: int = 1,
    ):
        # -------- mid & up --------
        x = self.conv_in(sample)                               # (B·F,128?,H,W)
        x = self.mid_block(x, image_only_indicator=image_only_indicator)
        x = x.cast(self.up_blocks[0].weight.dtype)             # 与上采样链 dtype 对齐

        for up in self.up_blocks:
            x = up(x, image_only_indicator=image_only_indicator)

        # -------- 输出分支 --------
        outputs = []
        B_F, C, H, W = x.shape
        B = B_F // num_frames

        for branch, tconv in zip(self.out_blocks, self.time_conv_outs):
            y = x
            for layer in branch:
                if isinstance(layer, SpatioTemporalResBlock):
                    y = layer(y, None, image_only_indicator)
                else:
                    y = layer(y)

            # (B·F,C,H,W) → (B,C,F,H,W) → 时域 3×1×1 卷积 → 拉回
            y = y.reshape(B, num_frames, *y.shape[1:]).permute(0,2,1,3,4)   # (B,C,F,H,W)
            y = tconv(y)
            y = y.permute(0,2,1,3,4).reshape(B_F, *y.shape[1:3], H, W)
            outputs.append(y)

        return outputs

class PMapAutoencoderKLTemporalDecoder(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 4,
        enc_down_block_types: Tuple[str] = (
            "DownEncoderBlock2D", "DownEncoderBlock2D",
            "DownEncoderBlock2D", "DownEncoderBlock2D"
        ),
        enc_block_out_channels: Tuple[int] = (128, 256, 512, 512),
        enc_layers_per_block: int = 2,
        dec_block_out_channels: Tuple[int] = (128, 256, 512, 512),
        dec_layers_per_block: int = 2,
        out_channels: Tuple[int] = (1, 1, 1),
        mid_block_add_attention: bool = True,
        offset_scale_factor: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=enc_down_block_types,
            block_out_channels=enc_block_out_channels,
            layers_per_block=enc_layers_per_block,
            double_z=False,
            mid_block_add_attention=mid_block_add_attention,
        )
        zero_module(self.encoder.conv_out)

        self.offset_scale_factor = offset_scale_factor

        self.decoder = PMapTemporalDecoder(
            in_channels=latent_channels,
            block_out_channels=dec_block_out_channels,
            layers_per_block=dec_layers_per_block,
            out_channels=out_channels,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, PMapTemporalDecoder)):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self) -> Dict[str, "AttentionProcessor"]:
        processors = {}

        def _collect(name: str, mod: nn.Module):
            if hasattr(mod, "get_processor"):
                processors[f"{name}.processor"] = mod.get_processor()
            for sub_name, child in mod.named_children():
                _collect(f"{name}.{sub_name}", child)

        for name, mod in self.named_children():
            _collect(name, mod)
        return processors

    def set_attn_processor(self,
                           processor: Union["AttentionProcessor",
                                            Dict[str, "AttentionProcessor"]]):
        count = len(self.attn_processors)
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"Expect {count} processors, got {len(processor)}."
            )

        def _set(name: str, mod: nn.Module, proc):
            if hasattr(mod, "set_processor"):
                if isinstance(proc, dict):
                    mod.set_processor(proc.pop(f"{name}.processor"))
                else:
                    mod.set_processor(proc)
            for sub, child in mod.named_children():
                _set(f"{name}.{sub}", child, proc)

        for name, mod in self.named_children():
            _set(name, mod, processor)

    def encode(self,
               x: jt.Var,
               latent_dist: "DiagonalGaussianDistribution"
               ) -> "DiagonalGaussianDistribution":
        h = self.encoder(x)
        offset = h * self.offset_scale_factor
        param  = latent_dist.parameters.cast(h.dtype)
        mean, logvar = jt.split(param, 2, dim=1)
        posterior = DiagonalGaussianDistribution(
            jt.concat([mean + offset, logvar], dim=1)
        )
        return posterior

    def decode(self,
               z: jt.Var,
               num_frames: int
               ) -> List[jt.Var]:
        B = z.shape[0] // num_frames
        img_only = jt.zeros((B, num_frames), dtype=z.dtype, device=z.device)
        return self.decoder(z, num_frames=num_frames,
                            image_only_indicator=img_only)
