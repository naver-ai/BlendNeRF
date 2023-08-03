# BlendNeRF
# Copyright (c) 2023-present NAVER Corp.
# This work is licensed under the NVIDIA Source Code License for EG3D.

import subprocess

for dataset in ['celeba_hq', 'afhq']:
    outdir = f'results/{dataset}'
    inversion_fname = 'inversion'

    if dataset == 'celeba_hq':
        network_path = 'checkpoints/original_ffhq_512-128_400kimg.pkl'
        encoder_path = 'checkpoints/encoder_ffhq.pkl'
        
        editing_parts = ['face', 'hair', 'nose', 'lip', 'eyes']
        enable_warping = False
        ref_color_lambda = 3 if editing_parts == 'hair' else 1
        
    elif dataset == 'afhq':
        network_path = 'checkpoints/afhqcats512-128.pkl'
        encoder_path = f'checkpoints/encoder_afhq.pkl'     

        editing_parts = ['face', 'ears', 'eyes']      
        enable_warping = True
        ref_color_lambda = 5

    original_img_path = f'test_images/{dataset}/original.png'
    reference_img_path = f'test_images/{dataset}/reference.png'

    cmd = f"python inversion.py --outdir={outdir}/{inversion_fname} --network={network_path} --encoder_network {encoder_path} --original_img_path {original_img_path} --reference_img_path {reference_img_path}"
    subprocess.run([cmd], shell=True, check=True)

    for editing_target in editing_parts:
        print("Blending", editing_target)

        for poisson, n_iterations in [[False, 200], [True, 100]]:
            generate_shape = not poisson
            cmd = f"python blend.py --outdir={outdir}/editing/{editing_target} --network={network_path} --editing_target {editing_target} --ref_color_lambda {ref_color_lambda} --inversion_path={outdir}/{inversion_fname}  --enable_warping {enable_warping} --shapes={generate_shape} --n_iterations={n_iterations} --poisson={poisson}"
            subprocess.run([cmd], shell=True, check=True)