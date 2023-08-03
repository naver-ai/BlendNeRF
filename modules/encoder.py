# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Discriminator architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import numpy as np
import torch
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from training.networks_stylegan2 import DiscriminatorBlock, Conv2dLayer, FullyConnectedLayer
from torch_utils import misc
from pytorch3d.transforms import euler_angles_to_matrix

@persistence.persistent_class
class EncoderEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        # cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        w_dim,                          # G w dim = 14
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        # mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        # mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        # self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture
        self.w_dim = w_dim

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        # self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp) #  in_channels + mbstd_num_channels
        
        # self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        
        latent_dim = 512
        self.latent = FullyConnectedLayer(in_channels, latent_dim)
        self.pose = FullyConnectedLayer(in_channels, 3) # 3: R 
        # or yaw pitch 2 + translation difference 3 = 5

    def forward(self, x, img):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        # if self.mbstd is not None:
        #     x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        # x = self.out(x)

        latent = self.latent(x)
        latent = latent.unsqueeze(1).repeat(1,self.w_dim,1) # G.w_dim
        pose = self.pose(x)

        # # Conditioning.
        # if self.cmap_dim > 0:
        #     misc.assert_shape(cmap, [None, self.cmap_dim])
        #     x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert latent.dtype == dtype
        assert pose.dtype == dtype

        return latent, pose

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


@persistence.persistent_class
class Encoder(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        # cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        # sr_upsample_factor  = 1,        # Ignored for SingleDiscriminator
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        # mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        # self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        # if cmap_dim is None:
        #     cmap_dim = channels_dict[4]
        # if c_dim == 0:
        #     cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        # if c_dim > 0:
        #     self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = EncoderEpilogue(channels_dict[4], resolution=4, **epilogue_kwargs, **common_kwargs)
        self.intrinsic = torch.tensor([0.0000,  0.0000,  0.0000,  1.0000, 4.2647,  0.0000,  0.5000,  0.0000,  4.2647,  0.5000,  0.0000,  0.0000, 1.0000]).unsqueeze(0) # FFHQ, AFHQ

    def forward(self, img, **block_kwargs):
        img = img['image']

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        latent, pose_euler = self.b4(x, img)
        R = euler_angles_to_matrix(pose_euler[:,:3], convention="XYZ") # N 3 3
        # R_t = torch.cat([R, pose[:,3:].unsqueeze(-1)], dim=2) # N 3 4
        T = -(R @ torch.tensor([0,0, 2.7], device=R.device)) # 1 3
        R_t = torch.cat([R, T.unsqueeze(-1)], dim=2) # N 3 4
        R_t = R_t.flatten(1) # N 12

        pose_all = torch.cat([R_t, self.intrinsic.to(R_t.device).repeat(R_t.shape[0], 1)], dim=1) # N 25

        return latent, pose_euler, pose_all

    def extra_repr(self):
        return f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------

def filtered_resizing(image_orig_tensor, size, f, filter_mode='antialiased'):
    if filter_mode == 'antialiased':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
    elif filter_mode == 'classic':
        ada_filtered_64 = upfirdn2d.upsample2d(image_orig_tensor, f, up=2)
        ada_filtered_64 = torch.nn.functional.interpolate(ada_filtered_64, size=(size * 2 + 2, size * 2 + 2), mode='bilinear', align_corners=False)
        ada_filtered_64 = upfirdn2d.downsample2d(ada_filtered_64, f, down=2, flip_filter=True, padding=-1)
    elif filter_mode == 'none':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False)
    elif type(filter_mode) == float:
        assert 0 < filter_mode < 1

        filtered = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        aliased  = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=False)
        ada_filtered_64 = (1 - filter_mode) * aliased + (filter_mode) * filtered
        
    return ada_filtered_64

