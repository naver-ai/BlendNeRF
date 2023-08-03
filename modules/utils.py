# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn
from modules.bisenet import BiSeNet
from modules.camera_utils import LookAtPoseSampler
import PIL.Image
import numpy as np
import torchvision
import scipy.interpolate
import imageio
from tqdm import tqdm
import cv2
from torchvision.utils import save_image
from torchvision import transforms

#----------------------------------------------------------------------------
def none_or_str(value):
    if value == 'None':
        return None
    else:
        return value

def get_cfg(network_pkl):
    if 'ffhq' in network_pkl.split('/')[-1]:
        cfg = 'ffhq'
    elif 'afhq' in network_pkl.split('/')[-1]:
        cfg = 'afhq'
    # elif 'shapenet' in network_pkl.split('/')[-1]:
    #     cfg = 'shapenet'

    return cfg

def get_conditioning_params(G, network_pkl, device): # front facing
    cfg = get_cfg(network_pkl)

    if cfg == 'shapenet':
        focal_length = 1.0254
        angle_ys = np.pi
    elif cfg in ['ffhq', 'afhq']:
        focal_length = 4.2647
        angle_ys = np.pi/2

    angle_p = np.pi/2

    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(angle_ys, angle_p, cam_pivot, radius=cam_radius, device=device)
    
    if cfg == 'shapenet':
        conditioning_cam2world_pose[:,:3,:] = torch.tensor([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]], device=device).unsqueeze(0) @ conditioning_cam2world_pose[:,:3,:]

    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    return conditioning_params

@torch.no_grad()
def generate_three_angles(G, network_pkl, ws, device, save_path, previous_imgs=None, warping_matrix=None):
    cfg = get_cfg(network_pkl)

    if cfg == 'shapenet':
        focal_length = 1.0254
        angle_p = np.pi/2 - 0.9
        angle_ys = [.6, 0, -.6]
        angle_ys = [yaw+np.pi for yaw in angle_ys]    
        cam_radius = 1.3    
    elif cfg in ['ffhq', 'afhq']:
        focal_length = 4.2647
        angle_p = np.pi/2 - 0.2
        angle_ys = [.4, 0, -.4]
        angle_ys = [yaw+np.pi/2 for yaw in angle_ys]
        cam_radius = 2.7

    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    # cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7) => it is wrong in ShapeNet Car

    if previous_imgs:
        imgs = previous_imgs
    else:
        imgs = []
    
    camera_params_list = []
    synthesized_img_list = []

    for angle_y in angle_ys:
        cam2world_pose = LookAtPoseSampler.sample(angle_y, angle_p, cam_pivot, radius=cam_radius, device=device)

        if cfg == 'shapenet': # different axis
            cam2world_pose[:,:3,:] = torch.tensor([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]], device=device).unsqueeze(0) @ cam2world_pose[:,:3,:] # [[0, 0, 1.0], [1, 0, 0], [0, 1, 0]]
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        camera_params_list.append(camera_params)

        img = G.synthesis(ws, camera_params, noise_mode='const', warping_matrix=warping_matrix)['image']
        synthesized_img_list.append(img)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs.append(img)

    imgs = torch.cat(imgs, dim=2)
    PIL.Image.fromarray(imgs[0].cpu().numpy(), 'RGB').save(save_path)

    return synthesized_img_list, camera_params_list

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class Segmentation_Network(nn.Module):
    def __init__(self, dataset, device='cuda'):
        super(Segmentation_Network, self).__init__()

        if dataset == 'ffhq': # test set is celeba_hq
            # BisetNet(semantic segmenation)
            # only 512x512-resolution image works for bisenet
            self.bisenet = BiSeNet('checkpoints/bisenet.ckpt').eval().to(device)
            requires_grad(self.bisenet, False)
            
            """
            | Label list | | |
            | ------------ | ------------- | ------------ |
            | 0: 'background' | 1: 'skin' | 2: 'r_eyebrow' |
            | 3: 'l_eyebrow' | 4: 'r_eye' | 5: 'l_eye' |
            | 6: 'glasses' | 7: 'r_ear' | 8: 'l_ear' |
            | 9: 'earring' | 10: 'nose' | 11: 'inner_lip' |
            | 12: 'upper_lip' | 13: 'lower_lip' | 14: 'neck' |
            | 15: 'necklace' | 16: 'cloth' | 17: 'hair' |
            | 18: 'hat' | | |
            """

        elif dataset == 'afhq':
            self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=16, aux_loss=None).eval().to(device)

            checkpoint = torch.load('checkpoints/deeplab_epoch_19.pth')
            self.deeplabv3.load_state_dict(checkpoint['model_state_dict'])

            requires_grad(self.deeplabv3, False)
            self.deeplab_res = 256

            self.resnet_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            
            """
            | Label list | | |
            | ------------ | ------------- | ------------ |
            | 7: 'skin' | 8: 'ears' | 9: 'eyes' |
            | 10: 'mouth' | 12: 'nose' | 
            """

            self.part_to_index = {
                'face': [7,9,10,12],
                'ears': [8],
                'eyes': [9],
                'mouth': [10],
                'nose': [12],
            }

        self.dataset = dataset
        self.device = device

    @torch.no_grad()
    def get_mask(self, src_imgs, ref_imgs, part):

        if self.dataset == 'ffhq':
            # face hair eye nose lip
            # face excludes ears
            
            src_masks = self.bisenet.get_masks(src_imgs)[part]
            ref_masks = self.bisenet.get_masks(ref_imgs)[part]
            src_masks[ref_masks==1] = 1 # overall mask

            if part == 'face':
                src_hair_masks = self.bisenet.get_masks(src_imgs)['hair']
                ref_hair_masks = self.bisenet.get_masks(ref_imgs)['hair']

                # remove hair part
                src_masks[src_hair_masks==1] = 0
                src_masks[ref_hair_masks==1] = 0

            return src_masks
        elif self.dataset == 'afhq':
            
            img_res = src_imgs.shape[-1]
            src_imgs = (src_imgs+1)/2 # -1~1 => 0~1
            assert len(src_imgs) == 1
            assert len(ref_imgs) == 1

            masks = []

            for img in [src_imgs, ref_imgs]:
                img = nn.functional.interpolate(img, size=self.deeplab_res, mode='bicubic')
                img = img[0] # remove batch dim
                img = self.resnet_transform(img).unsqueeze(0)
                y_pred = self.deeplabv3(img)['out']
                y_pred = torch.log_softmax(y_pred, dim=1)
                _, y_pred = torch.max(y_pred, dim=1)

                mask = torch.zeros(y_pred.shape, device=self.device)
                for i in self.part_to_index[part]:
                    mask[y_pred==i] = 1
                masks.append(mask)
            
            src_mask = masks[0]
            ref_mask = masks[1]
            src_mask[ref_mask==1] = 1 # 1, 256, 256

            target_mask = src_mask.unsqueeze(0) # 1, 1, 256, 256
            target_mask = nn.functional.interpolate(target_mask, size=img_res, mode='nearest')

            return target_mask         


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

@torch.no_grad()
def gen_interp_video(G, mp4: str, latent_c, mesh_visualize=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, cfg='FFHQ', image_mode='image', device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(latent_c) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(latent_c) // (grid_w*grid_h)

    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    focal_length = 4.2647 if cfg != 'shapenet' else 1.7074 # shapenet has higher FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(latent_c), 1)
    ws = latent_c # N 14 512
    _ = G.synthesis(ws[:1], c[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)

    all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.25
                yaw_range = 0.35
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                        3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                        camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                focal_length = 4.2647 if cfg != 'shapenet' else 1.7074 # shapenet has higher FOV
                intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')[image_mode][0]

                if mesh_visualize is not None:
                    if xi == 0:
                        mesh_path = mesh_visualize['mesh_paths']['original']
                    elif xi == 1:
                        mesh_path = mesh_visualize['mesh_paths']['reference']
                    elif xi == 2:
                        mesh_path = mesh_visualize['mesh_paths']['blending']
                    

                    mesh_img = mesh_visualize['mesh_ext'].render_mesh(c[0:1], mesh_path).squeeze(0)
                    img = torch.cat([img, mesh_img], dim=1)
                imgs.append(img)
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()

@torch.no_grad()
def save_img_w_mask(img, mask, outdir, img_name, device):
    img = img.detach().clone()
    mask_squeeze = mask[0].permute(1, 2, 0) # 1 3 H W => H W 3
    mask_squeeze_nd = (mask_squeeze * 255).clamp(0, 255).to(torch.uint8).cpu().numpy() # H W 3
    mask_gray = mask_squeeze_nd[:,:,0] # H W 3 => H W
    _, mask_gray = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    # Draw contours:
    border_thickness = 10
    mask_using_contour = cv2.drawContours(mask_squeeze_nd, contours, -1, (255, 255, 255), border_thickness) # red
    # # Draw original mask inside contours
    mask_using_contour = cv2.drawContours(mask_squeeze_nd, contours, -1, (0, 0, 0), -1)  # black
    mask_using_contour = transforms.ToTensor()(mask_using_contour).to(device)

    # draw red
    img[:,0][mask_using_contour == 1] = 1
    img[:,1][mask_using_contour == 1] = -1
    img[:,2][mask_using_contour == 1] = -1

    save_image(img, f'{outdir}/{img_name}_w_mask.png', normalize=True, range=(-1,1)) 
