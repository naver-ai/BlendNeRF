# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import click
import dnnlib
import torch
from modules.legacy import load_network_pkl
import lpips
from tqdm import tqdm
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from torch.nn import functional as F
from torchvision.utils import save_image
from modules.utils import generate_three_angles, Segmentation_Network, requires_grad, get_cfg, none_or_str, gen_interp_video, save_img_w_mask
from modules.mesh_renderer import Mesh_extractor
from modules.poisson_blending import get_mask_for_PB, save_result_PB
torch.manual_seed(0) # for fixed const noise for the generator
torch.set_printoptions(precision=2)

#----------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--editing_target', help='Where to edit the images', type=str, default='face')
@click.option('--n_iterations', type=int, required=False, default=200, show_default=True)
@click.option('--save_period', type=int, required=True, default=50, show_default=True)
@click.option('--inversion_path', help='inversion folder path', type=none_or_str)
@click.option('--ref_color_lambda', help='ref_color_lambda', type=int, default=1)
@click.option('--enable_warping', help='warp the reference', type=bool, required=True)
@click.option('--poisson', help='ours with Poisson blending', type=bool, required=True)
def generate_images(
    network_pkl: str,
    outdir: str,
    shapes: bool,
    editing_target: str,
    n_iterations: int,
    save_period: int,
    inversion_path: str,
    ref_color_lambda: int,
    enable_warping: bool,
    poisson: bool,
):  
    outdir = os.path.join(outdir, f'poisson_{poisson}')

    os.makedirs(outdir, exist_ok=True)

    
    print('Loading initial networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # reload_modules
    print("Reloading Modules!")
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    print('Loading networks from "%s"...' % inversion_path)
    inversion_model_path = os.path.join(inversion_path, 'G_optimized.pt')
    G.load_state_dict(torch.load(inversion_model_path))

    cfg = get_cfg(network_pkl)
    img_size = G.rendering_kwargs.get('image_resolution')

    shape_format = '.ply'
    ray_resolution = 128

    if cfg == 'shapenet':
        shape_res = 64
        assert img_size == 128
    else: # afhq, ffhq
        img_size = G.rendering_kwargs.get('image_resolution')
        shape_res = 128
        assert img_size == 512
    
    mesh_ext = Mesh_extractor(G, shape_res, shape_format, ray_resolution, device=device) # for local alignment
    l1_loss = torch.nn.L1Loss()
    noise_mode = 'const'

    ##################
    # load inversion #
    ##################

    image_info_path = os.path.join(inversion_path, 'images.pt')
    image_info = torch.load(image_info_path)
      
    latent_original = image_info['original']['w_pivot']
    latent_reference = image_info['reference']['w_pivot']  

    #########################
    # blending start!!
    #########################
    if poisson:
        target_latent = latent_reference.clone().detach()
    else:
        target_latent = latent_original.clone().detach()
    target_latent.requires_grad_()
    direction_optimizer = torch.optim.Adam([target_latent], lr=1e-2)

    percept = lpips.LPIPS(net='vgg').to(device) 
    percept.eval()
    segnet = Segmentation_Network(dataset=cfg)
    requires_grad(G, False)
    requires_grad(percept, False)
    requires_grad(segnet, False)

    # visualization purpose only: save img with mask
    for img_name in ['reference', 'original']:
        img = image_info[img_name]['img']
        save_img_w_mask(img, segnet.get_mask(img, img, part=editing_target), outdir, img_name, device=device)

    masks_for_align = {}
    ply_mesh_paths = {}

    # Mask should be given in the original img's camera pose
    camera_original = image_info['original']['camera'] 

    for img_name, latent_code in zip(['reference', 'original'], [latent_reference, latent_original]):
        ply_mesh_path = os.path.join(outdir, f'mesh_{img_name}.ply')
        ply_mesh_paths[img_name] = ply_mesh_path
        if not os.path.exists(ply_mesh_path):
            mesh_ext.save_shape(latent_code, save_path=ply_mesh_path)

        # get masks using the pretrained network
        img_recon = G.synthesis(latent_code, camera_original, noise_mode=noise_mode)['image']
        masks_for_align[img_name] = segnet.get_mask(img_recon, img_recon, part=editing_target)    

        if enable_warping == True:
            try:
                masks_for_align[img_name] = F.interpolate(masks_for_align[img_name], size=(ray_resolution, ray_resolution)) 
                mesh_ext.projection(ply_mesh_paths[img_name], camera_original, mask_name=editing_target, mask=masks_for_align[img_name])
            except: # mask back-projection failed.
                print("Initialization steps of local alignment failed. We're going to apply global alignment only.")
                enable_warping = False
    
    if enable_warping == False: # no warping
        warping_matrix = torch.eye(4, device=device, dtype=torch.float32)
    else:
        try:
            warping_matrix, avg_scale, translation_distance = mesh_ext.get_warping_matrix(ply_mesh_paths['original'], ply_mesh_paths['reference'], editing_target, save_result=f'{outdir}/icp_result.png')
            print('avg_scale',avg_scale,'translation_distance',translation_distance)

            warping_matrix[0,0] = torch.clamp(warping_matrix[0,0], min=0.75, max=1.25)
            warping_matrix[1,1] = torch.clamp(warping_matrix[1,1], min=0.75, max=1.25)
            warping_matrix[2,2] = torch.clamp(warping_matrix[2,2], min=0.75, max=1.25)
        except: # Library issues or ICP sometimes fails
            print("Local alignment failed. We're going to apply global alignment only.")
            enable_warping = False
            warping_matrix = torch.eye(4, device=device, dtype=torch.float32)
    print('warping_matrix: ', warping_matrix)

    with torch.no_grad():
        original_densities_coarse = G.synthesis(latent_original, camera_original, noise_mode=noise_mode)['densities_coarse']
        original_img = image_info['original']['img']
        aligned_ref = G.synthesis(latent_reference, camera_original, noise_mode=noise_mode, warping_matrix=warping_matrix)
        ref_img, ref_densities_coarse = aligned_ref['image'], aligned_ref['densities_coarse']
        mask = segnet.get_mask(original_img, ref_img, part=editing_target) # mask for editing using aligned images
        mask_resized_shape_res = F.interpolate(mask, size=(shape_res, shape_res)) # density
        save_image(mask, f'{outdir}/mask.png')
        save_image(ref_img, f'{outdir}/ref_aligned.png', normalize=True, range=(-1,1)) 

    variation_weight = (shape_res**2) / torch.count_nonzero(mask_resized_shape_res)
    
    if poisson: 
        mask_using_contour, mask_for_PB, original_FG_mask = get_mask_for_PB(mask, editing_target, outdir, device)
    
    for i in tqdm(range(0, n_iterations+1)):
        if poisson: 
            # starts from the warped reference image
            edited = G.synthesis(target_latent, camera_original, noise_mode=noise_mode, warping_matrix=warping_matrix)
            edited_image, edited_densities_coarse = edited['image'], edited['densities_coarse']
            src_rgb_loss = 10 * l1_loss(original_img*mask_using_contour, edited_image*mask_using_contour)
            src_lpips_loss = 10 * percept(original_img*mask_using_contour, edited_image*mask_using_contour)
        else:
            # starts from the original image
            edited = G.synthesis(target_latent, camera_original, noise_mode=noise_mode)
            edited_image, edited_densities_coarse = edited['image'], edited['densities_coarse']
            src_rgb_loss = l1_loss(original_img*(1-mask), edited_image*(1-mask))
            src_lpips_loss = percept(original_img*(1-mask), edited_image*(1-mask))

        rgb_loss = (src_rgb_loss).mean()
        ref_lpips_loss = 0.1 * ref_color_lambda * variation_weight * percept(ref_img*mask, edited_image*mask)
        lpips_loss  = (src_lpips_loss + ref_lpips_loss).mean()

        src_depth_loss = l1_loss(original_densities_coarse*(1-mask_resized_shape_res), edited_densities_coarse*(1-mask_resized_shape_res))
        ref_depth_loss = variation_weight * (l1_loss(ref_densities_coarse*mask_resized_shape_res, edited_densities_coarse*mask_resized_shape_res))
        depth_loss = (src_depth_loss + ref_depth_loss).mean()

        loss = depth_loss + 10 * (lpips_loss + rgb_loss)
        loss.backward()

        direction_optimizer.step()
        direction_optimizer.zero_grad()  

        if i % save_period == 0:
            src_rgb_loss_val = src_rgb_loss.item() if torch.is_tensor(src_rgb_loss) else 0 
            src_lpips_loss_val = src_lpips_loss.item() if torch.is_tensor(src_lpips_loss) else 0 
            ref_lpips_loss_val = ref_lpips_loss.item() if torch.is_tensor(ref_lpips_loss) else 0 
            src_depth_loss_val = src_depth_loss.item() if torch.is_tensor(src_depth_loss) else 0 
            ref_depth_loss_val = ref_depth_loss.item() if torch.is_tensor(ref_depth_loss) else 0 
        
            print('src_rgb %.1f src_lpips %.1f ref_lpips %.1f src_depth %.1f ref_depth %.1f' % (src_rgb_loss_val, src_lpips_loss_val, ref_lpips_loss_val, src_depth_loss_val, ref_depth_loss_val))

            if poisson:
                save_result_PB(original_img, edited_image, editing_target, mask_for_PB, original_FG_mask, outdir, i)
            else:
                generate_three_angles(G, network_pkl, target_latent, device, f'{outdir}/edited_multiview_{i}.png')
                save_image(edited_image, f'{outdir}/edited_{i}.png', normalize=True, range=(-1,1))

    target_latent = target_latent.requires_grad_(False)
    result = {'target_latent': target_latent}
    torch.save(result, f'{outdir}/result.pt')

    if not poisson:
        if shapes: # save mesh to visualize the results
            shape_res_for_result = 512 #512
            shape_format_for_result = '.ply' # .mrc
            os.makedirs(f'{outdir}/mesh_results', exist_ok=True)
            mesh_visualize = {
                'mesh_ext': Mesh_extractor(G, shape_res_for_result, shape_format_for_result, ray_resolution, device=device),
                'mesh_paths': {
                    'original': f'{outdir}/mesh_results/mesh_original_{shape_res_for_result}{shape_format_for_result}',
                    'reference': f'{outdir}/mesh_results/mesh_reference_{shape_res_for_result}{shape_format_for_result}',
                    'blending': f'{outdir}/mesh_results/mesh_blending_{shape_res_for_result}{shape_format_for_result}'
                }
            }
            mesh_visualize['mesh_ext'].save_shape(latent_original, save_path=mesh_visualize['mesh_paths']['original'])
            mesh_visualize['mesh_ext'].save_shape(latent_reference, save_path=mesh_visualize['mesh_paths']['reference'])
            mesh_visualize['mesh_ext'].save_shape(target_latent, save_path=mesh_visualize['mesh_paths']['blending'])
        else:
            mesh_visualize = None

        latent_codes = [latent_original, latent_reference, target_latent]
        latent_codes = torch.cat(latent_codes, dim=0)

        gen_interp_video(G=G, mp4=f'{outdir}/video_blend.mp4', latent_c=latent_codes, bitrate='10M', grid_dims=(3,1), w_frames=120, cfg=cfg) # if you want to render meshes, pass mesh_visualize to the func.

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()

#----------------------------------------------------------------------------
