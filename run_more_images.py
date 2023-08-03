# BlendNeRF
# Copyright (c) 2023-present NAVER Corp.
# This work is licensed under the NVIDIA Source Code License for EG3D.

import subprocess

for dataset in ['celeba_hq', 'afhq']:
    if dataset == 'celeba_hq':
        network_path = 'checkpoints/original_ffhq_512-128_400kimg.pkl'
        encoder_path = 'checkpoints/encoder_ffhq.pkl'
        
        editing_parts = ['face', 'nose', 'eyes', 'lip', 'hair']
        target_fnames = ['83_126', '102_314', '107_245', '135_434', '116_27']
        enable_warping = False
        ref_color_lambda = 3 if editing_parts == 'hair' else 1
        
    elif dataset == 'afhq':
        network_path = 'checkpoints/afhqcats512-128.pkl'
        encoder_path = f'checkpoints/encoder_afhq.pkl'     

        editing_parts = ['face', 'ears', 'eyes', 'ears', 'face']
        target_fnames = ['2_325', '171_322', '173_210', '412_31', '442_154']
        enable_warping = True
        ref_color_lambda = 5

    poisson = False
    n_iterations = 200
    generate_shape = not poisson

    for fname, editing_target in zip(target_fnames, editing_parts):
        outdir = f'results/{dataset}/{fname}'
        inversion_fname = 'inversion'
        original_img_path = f'test_images/{dataset}/{fname}_{editing_target}/original.png'
        reference_img_path = f'test_images/{dataset}/{fname}_{editing_target}/reference.png'
        
        cmd = f"python inversion.py --outdir={outdir}/{inversion_fname} --trunc=0.7 --network={network_path} --encoder_network {encoder_path} --original_img_path {original_img_path} --reference_img_path {reference_img_path}"
        subprocess.run([cmd], shell=True, check=True)

        cmd = f"python blend.py --outdir={outdir}/editing/{editing_target} --network={network_path} --editing_target {editing_target} --ref_color_lambda {ref_color_lambda} --inversion_path={outdir}/{inversion_fname}  --enable_warping {enable_warping} --shapes={generate_shape} --n_iterations={n_iterations} --poisson={poisson}"
        subprocess.run([cmd], shell=True, check=True)