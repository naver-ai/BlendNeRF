# Refer to PTI. https://github.com/danielroich/PTI

"""Generate images and shapes using pretrained network pickle."""

import os
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from modules.legacy import load_network_pkl
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from modules.utils import generate_three_angles, get_conditioning_params
# from pytorch3d.transforms import euler_angles_to_matrix
import torchvision.transforms as transforms
from torchvision.utils import save_image

#----------------------------------------------------------------------------

import lpips

np.random.seed(0)
torch.manual_seed(0)

#----------------------------------------------------------------------------

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

#----------------------------------------------------------------------------

def get_morphed_w_code(new_w_code, fixed_w):
    regulizer_alpha = 30
    interpolation_direction = new_w_code - fixed_w
    interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
    direction_to_move = regulizer_alpha * interpolation_direction / interpolation_direction_norm
    result_w = fixed_w + direction_to_move

    return result_w

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--encoder_network', 'encoder_network_pkl', help='encoder_network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--w_n_iterations', type=int, required=True, default=300, show_default=True)
@click.option('--g_n_iterations', type=int, required=True, default=100, show_default=True)
@click.option('--save_period', type=int, required=True, default=100, show_default=True)
@click.option('--original_img_path', type=str, required=True)
@click.option('--reference_img_path', type=str, required=True)
def invert_images(
    network_pkl: str,
    encoder_network_pkl: str,
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    w_n_iterations: int,
    g_n_iterations: int,
    save_period:int,
    original_img_path: int,
    reference_img_path: int,
):
    os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = load_network_pkl(f)['G_ema'].to(device) # type: ignore
    
    print('Loading networks from "%s"...' % encoder_network_pkl)
    with dnnlib.util.open_url(encoder_network_pkl) as f:
        E = load_network_pkl(f)['E'].eval().requires_grad_(False).to(device) # type: ignore
        
    # reload_modules
    print("Reloading Modules!")
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    # for reg
    G_original = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_original, require_all=True)
    G_original.neural_rendering_resolution = G.neural_rendering_resolution
    G_original.rendering_kwargs = G.rendering_kwargs    

    params_g = []
    params_dict_g = dict(G.named_parameters())
    for _, value in params_dict_g.items():
        params_g += [{'params':[value], 'lr': 0.0025}]
        
    g_optim = torch.optim.Adam(params_g, betas=(0, 0.9))
    requires_grad(G, False)

    lpips_loss = lpips.LPIPS(net='vgg').eval().requires_grad_(False).to(device)
    l2_loss = torch.nn.MSELoss()
    noise_mode = 'const'

    #####################
    # load image
    #####################

    to_tensor = transforms.ToTensor()
    images = {}
    img_size = 512
    for img_name, img_path in zip(['original', 'reference'], [original_img_path, reference_img_path]):
        img = PIL.Image.open(img_path)
        img = img.resize((img_size, img_size))
        img = to_tensor(img).to(device).unsqueeze(0)
        img = (img - 0.5) * 2
        
        images[img_name] = {
            'img': img
        }

    #####################
    # Optimize W
    #####################
    
    for img_name in images.keys():
        # w_pivot_save_path = f'{outdir}/{img_name}_w_pivot.pt'
        img = images[img_name]['img']
    
        # get the pose label using the pretrained encoder
        _, _, estimated_pose = E({'image': img})
        images[img_name]['camera'] = estimated_pose

        w_pivot = G.backbone.mapping.w_avg.clone().detach() 
        w_pivot = w_pivot.requires_grad_(True)
        direction_optimizer = torch.optim.Adam([w_pivot], lr=1e-2)

        for i in tqdm(range(w_n_iterations)): # n_iterations + 1
            w_plus = w_pivot.unsqueeze(0).unsqueeze(1).repeat(1,14,1) # 512 -> 1 14 512
            inverted_img = G.synthesis(w_plus, estimated_pose, noise_mode=noise_mode)['image']
            loss = l2_loss(img, inverted_img).mean()
            loss += lpips_loss(img, inverted_img).mean()

            direction_optimizer.zero_grad()  
            loss.backward()
            direction_optimizer.step()

            if i % save_period == 0:
                print(loss.item(), 'w_pivot optimization loss')
                save_image(inverted_img, f'{outdir}/{img_name}_inverted_W_{i}.png', normalize=True, range=(-1,1))
                generate_three_angles(G, network_pkl, w_plus, device, f'{outdir}/{img_name}_inverted_W_multiview_{i}.png')
            
        w_pivot = w_pivot.requires_grad_(False)
        images[img_name]['w_pivot'] = w_pivot.unsqueeze(0).unsqueeze(0).repeat(1,14,1) # save as W+ format => 1 14 512
        torch.save(images, f'{outdir}/images.pt')
        save_image(inverted_img, f'{outdir}/{img_name}_inverted_W.png', normalize=True, range=(-1,1))
        generate_three_angles(G, network_pkl, w_plus, device, f'{outdir}/{img_name}_inverted_W_multiview.png')
  
    #####################
    # Optimize G
    #####################

    conditioning_params= get_conditioning_params(G, network_pkl, device)
    requires_grad(G, True)
    reg_lambda = 0.1

    for i in tqdm(range(g_n_iterations)):

        for img_name in images.keys():
            original_img = images[img_name]['img']
            w_pivot = images[img_name]['w_pivot']
            estimated_pose = images[img_name]['camera']

            inverted_img = G.synthesis(w_pivot, estimated_pose, noise_mode=noise_mode)['image']
            loss = l2_loss(original_img, inverted_img).mean()
            loss += lpips_loss(original_img, inverted_img).mean()

            # for reg
            with torch.no_grad():
                z_samples = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(device)
                w_samples = G_original.mapping(z_samples, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                assert len(w_samples) == 1
                assert len(w_pivot) == 1
            
            territory_indicator_ws = get_morphed_w_code(w_samples, w_pivot)
            new_object = G.synthesis(territory_indicator_ws, estimated_pose, noise_mode=noise_mode)
            new_img = new_object['image']
            with torch.no_grad():
                ori_object = G_original.synthesis(territory_indicator_ws, estimated_pose, noise_mode=noise_mode)
                ori_img = ori_object['image']

            loss += l2_loss(new_img, ori_img).mean() * reg_lambda
            loss += lpips_loss(new_img, ori_img).mean() * reg_lambda

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

            with torch.no_grad():
                if i % save_period == 0:
                    print(loss.item(), 'G optimization loss')
                    save_image(inverted_img, f'{outdir}/{img_name}_inverted_G_{i}.png', normalize=True, range=(-1,1))
                    generate_three_angles(G, network_pkl, w_pivot, device, f'{outdir}/{img_name}_inverted_G_multiview_{i}.png')
                elif i + 1 == g_n_iterations:
                    save_image(inverted_img, f'{outdir}/{img_name}_inverted_G.png', normalize=True, range=(-1,1))
                    generate_three_angles(G, network_pkl, w_pivot, device, f'{outdir}/{img_name}_inverted_G_multiview.png')

    torch.save(G.state_dict(), f'{outdir}/G_optimized.pt')
#----------------------------------------------------------------------------

if __name__ == "__main__":
    invert_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
