import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
import segmentation_models_pytorch as smp


class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
    ):
        super(UNetModel, self).__init__()
        self.model = smp.Unet('efficientnet-b7', classes=3, encoder_weights='imagenet', encoder_depth=11, decoder_channels=[896, 896, 672, 672, 672, 448, 448, 448, 224, 224, 224])


    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        x = self.model(x)
        return x