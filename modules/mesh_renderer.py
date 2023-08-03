# BlendNeRF
# Copyright (c) 2023-present NAVER Corp.
# This work is licensed under the NVIDIA Source Code License for EG3D.
# Copy partial codes from EG3D: create_samples, save_shape. Others are our contributions.

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Refer to below documents.
# https://en.wikipedia.org/wiki/Euler_angles
# https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html

# Pytorch3d camera and world coordinates
# https://pytorch3d.org/docs/renderer_getting_started 
# https://pytorch3d.org/docs/cameras

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
import trimesh # /opt/conda/envs/blendnerf/lib/python3.9/site-packages/trimesh
from modules.modified_trimesh import icp
from modules.shape_utils import convert_sdf_samples_to_ply
from pytorch3d.structures import Meshes
from pytorch3d.io.ply_io import load_ply
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer, MeshRasterizer, PointLights, HardPhongShader
    )
from training.volumetric_rendering.ray_sampler import RaySampler
import matplotlib.pyplot as plt
import mrcfile
from tqdm import tqdm

class Mesh_extractor():
    def __init__(self, G, shape_res, shape_format, ray_resolution, device='cuda'):
        super(Mesh_extractor, self).__init__()
        self.G = G
        self.G.renderer.plane_axes = G.renderer.plane_axes.to(device)
        self.img_size = self.G.rendering_kwargs.get('image_resolution')
        self.shape_res = shape_res
        self.shape_format = shape_format
        self.cube_length = G.rendering_kwargs['box_warp'] # 1.0
        self.device = device
        self.lights = PointLights(device=self.device, location=[[-5.0, 0.0, 0.0]], ambient_color=((0.6, 0.6, 0.6),))
        self.ray_resolution = ray_resolution

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters.
        self.raster_settings = RasterizationSettings(
            image_size=self.shape_res,
            blur_radius=0,
            faces_per_pixel=1,
            # perspective_correct=False
        )

        self.R_camera_eg2py3d = torch.tensor([[-1, 0, 0], [0, -1, 0], [0,0,1]], device=self.device, dtype=torch.float32).unsqueeze(0)
        self.R_world_eg2py3d = torch.tensor([[0, 0, 1], [0, 1, 0], [-1,0,0]], device=self.device, dtype=torch.float32).unsqueeze(0)

        self.mesh_info = {}

        if G.rendering_kwargs['avg_camera_radius'] == 2.7: # ffhq, afhq
            self.focal_length = 4.2647
            self.cam_radius = 2.7
        else: # shapenet car
            self.focal_length = 1.0254
            self.cam_radius = 1.3

    # https://github.com/NVlabs/eg3d/blob/7cf1fd1e99e1061e8b6ba850f91c94fe56e7afe4/eg3d/gen_samples.py#L79
    def create_samples(self, N=512, voxel_origin=[0, 0, 0]):
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = np.array(voxel_origin) - self.cube_length/2
        voxel_size = self.cube_length / (N - 1)
        # -1,-1,-1 ~ 1,1,1 cube

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 3)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.float() / N) % N
        samples[:, 0] = ((overall_index.float() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        return samples.unsqueeze(0), voxel_origin, voxel_size

    # refer to https://github.com/NVlabs/eg3d/blob/7cf1fd1e99e1061e8b6ba850f91c94fe56e7afe4/eg3d/gen_samples.py#L184
    def save_shape(self, latent, save_path=None, trim_border=True): #  latent_type='z', 
        # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
        max_batch=1000000

        samples, _, _ = self.create_samples(N=self.shape_res, voxel_origin=[0, 0, 0])
        samples = samples.to(self.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=self.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=self.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = self.G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], latent, noise_mode='const')['sigma']
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((self.shape_res, self.shape_res, self.shape_res)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)

        # Trim the border of the extracted cube
        if trim_border:
            pad = int(30 * self.shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value
        
        sigmas = np.transpose(sigmas, (2, 1, 0))
        if (save_path != None) and (self.shape_format == '.ply'):
            convert_sdf_samples_to_ply(sigmas, [-self.cube_length/2, -self.cube_length/2, -self.cube_length/2], self.cube_length/(self.shape_res-1), save_path, level=10)
        elif self.shape_format == '.mrc': # output mrc
            with mrcfile.new_mmap(save_path, overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                mrc.data[:] = sigmas

        return sigmas

    @torch.no_grad()
    def render_mesh(self, camera_pose_all, ply_mesh_path):
        verts, faces = load_ply(ply_mesh_path)
        verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb.to(self.device))
        meshes = Meshes(verts=[verts], faces=[faces], textures=textures).to(self.device)
        camera_pose_extrinsics = camera_pose_all[:,:16].view(-1,4,4) # 1,4,4

        # rotation, translation are in eg3d coordinate
        rotation = camera_pose_extrinsics[:,:3,:3] # 1,3,3
        R = self.R_world_eg2py3d @ rotation @ torch.inverse(self.R_camera_eg2py3d) 

        dataset_translation = camera_pose_extrinsics[:,:3,3]
        # translation difference vector in eg3d world coordinate -> eg3d camera coordinate -> pytorch3d camera coordinate
        T = self.R_camera_eg2py3d @ torch.inverse(rotation) @ -dataset_translation.unsqueeze(-1)
        T = T.squeeze(-1) 

        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=2*np.arctan((1.0/2)/self.focal_length)*180/np.pi) # 13.37...
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            shader=HardPhongShader(device=self.device, cameras=cameras, lights=self.lights)
        )

        # Render Meshes object
        image = renderer(meshes)
        image = image[...,:3]      
        image = (image - 0.75)*4 # 0.5~1.0 -> -1~1
        image = image.permute(0,3,1,2)

        return image 

    @torch.no_grad()
    def projection(self, ply_mesh_path, camera, mask_name, mask):
        mask = mask.squeeze()
        # mask to world 3D points
        cam2world_matrix = camera[:, :16].view(-1, 4, 4)
        intrinsics = camera[:, 16:25].view(-1, 3, 3)    
        ray_sampler = RaySampler()

        ray_origins, ray_directions = ray_sampler(cam2world_matrix, intrinsics, self.ray_resolution) 
        ray_origins = ray_origins.squeeze(0)
        ray_directions = ray_directions.squeeze(0)

        assert (mask.shape[0] * mask.shape[1]) == ray_origins.shape[0]

        ray_origins_selected, ray_directions_selected = torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        mask_reshape = mask.permute(0, 1)
        mask_reshape = mask_reshape.flatten() # H W => H*W
        for i in range(len(mask_reshape)):
            if mask_reshape[i] == 1.0:
                ray_origins_selected = torch.cat([ray_origins_selected, ray_origins[i:i+1, :]])
                ray_directions_selected = torch.cat([ray_directions_selected, ray_directions[i:i+1, :]])

        R_world_eg2py3d = self.R_world_eg2py3d.squeeze(0)
        ray_origins_selected = R_world_eg2py3d @ ray_origins_selected.permute(1, 0)
        ray_directions_selected = R_world_eg2py3d @ ray_directions_selected.permute(1, 0)
        ray_origins_selected = ray_origins_selected.permute(1,0)
        ray_directions_selected = ray_directions_selected.permute(1,0)
        ray_origins_selected = ray_origins_selected.cpu()
        ray_directions_selected = ray_directions_selected.cpu()

        if ply_mesh_path in self.mesh_info.keys():
            trimesh_mesh = self.mesh_info[ply_mesh_path]['trimesh_mesh']
        else:
            trimesh_mesh = trimesh.load_mesh(ply_mesh_path)
            self.mesh_info[ply_mesh_path] = {
                'trimesh_mesh': trimesh_mesh
            }

        self.mesh_info[ply_mesh_path][mask_name] = {
            'initial_mask': mask
        }

        locations, index_ray, index_tri = trimesh_mesh.ray.intersects_location(
        ray_origins=ray_origins_selected,
        ray_directions=ray_directions_selected, multiple_hits=False) # only use first intersection

        self.mesh_info[ply_mesh_path][mask_name].update({
            'locations': locations,
            'index_ray': index_ray,
            'index_tri': index_tri
        })

        return

    @torch.no_grad()
    def get_warping_matrix(self, ori_ply_mesh_path, ref_ply_mesh_path, mask_name, save_result=None):
        ori_locations = self.mesh_info[ori_ply_mesh_path][mask_name]['locations']
        ref_locations = self.mesh_info[ref_ply_mesh_path][mask_name]['locations']
        R_world_py3d2eg = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1,0,0, 0], [0,0,0,1]], dtype=np.float) 

        total_matrix, transformed, cost = icp(ref_locations, ori_locations, reflection=False,
               translation=True,
               scale=True, uniform_scale=True, rotation=False, threshold=1e-6)      
        total_matrix = R_world_py3d2eg @ total_matrix @ np.transpose(R_world_py3d2eg)

        if save_result is not None: #save_result:
            fig = plt.figure(figsize=(12,7))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(ori_locations[:, 0], ori_locations[:, 1], ori_locations[:, 2], c='red', alpha=0.2)
            ax.scatter(ref_locations[:, 0], ref_locations[:, 1], ref_locations[:, 2], c='lightblue')
            ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c='blue')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.savefig(save_result)

        total_matrix = torch.from_numpy(total_matrix).type(torch.FloatTensor).to(self.device)
        avg_scale = torch.trace(total_matrix[:3,:3])/3 
        translation_distance = torch.norm(total_matrix[:3,3], p=2)

        return total_matrix, avg_scale, translation_distance