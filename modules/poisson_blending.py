# BlendNeRF
# Copyright (c) 2023-present NAVER Corp.
# This work is licensed under the NVIDIA Source Code License for EG3D.

import torch
import numpy as np
import cv2
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

def get_mask_for_PB(mask, editing_target, outdir, device):
    mask_squeeze = mask[0].permute(1, 2, 0) # 1 3 H W => H W 3
    mask_squeeze_nd = (mask_squeeze * 255).clamp(0, 255).to(torch.uint8).cpu().numpy() # H W 3
    mask_gray = mask_squeeze_nd[:,:,0] # H W 3 => H W
    _, mask_gray = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw contours:
    border_thickness = 30 if editing_target in ['eyes', 'lip'] else 50
    mask_using_contour = cv2.drawContours(mask_squeeze_nd, contours, -1, (255, 255, 255), border_thickness) # white
    # Draw original mask inside contours
    mask_using_contour = cv2.drawContours(mask_squeeze_nd, contours, -1, (0, 0, 0), -1)  # black
    mask_using_contour = transforms.ToTensor()(mask_using_contour).unsqueeze(0).to(device)
    save_image(mask_using_contour, f'{outdir}/mask_using_contour.png') # cv2.imwrite(f'{outdir}/mask_using_contour.png', mask_using_contour)

    total_mask_include_contour = mask_squeeze.clone() #  H W 3 # print(total_mask_include_contour.shape, mask_using_contour.shape) # 512 512 1 / 1 1 512 512
    total_mask_include_contour[mask_using_contour[0][0].unsqueeze(-1)==1] = 1

    # diffuse
    save_image(total_mask_include_contour.permute(2, 0, 1), f'{outdir}/total_mask_include_contour.png') # 
    mask_for_PB = (total_mask_include_contour * 255).clamp(0, 255).to(torch.uint8).cpu().numpy() # H W C / 0 ~ 255

    if editing_target == 'hair': # prevent ref color disappearing
        original_FG_mask =  mask_squeeze.clone()
        original_FG_mask = (original_FG_mask * 255).clamp(0, 255).to(torch.uint8).cpu().numpy() # H W C / 0 ~ 255
    else:
        original_FG_mask = None
        
    return mask_using_contour, mask_for_PB, original_FG_mask


def save_result_PB(original_img, edited_image, editing_target, mask_for_PB, original_FG_mask, outdir, step_i):
    original_img_PB = (original_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy() # H W C 0 255
    editied_img_PB = (edited_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy() # H W C 0 255

    if editing_target == 'hair': # prevent ref color disappearing
        original_img_PB[np.repeat(original_FG_mask, 3, axis=2)>127] = editied_img_PB[np.repeat(original_FG_mask, 3, axis=2)>127]

    br = cv2.boundingRect(mask_for_PB) # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2 , br[1] + br[3] // 2 )
    im_clone = cv2.seamlessClone(editied_img_PB, original_img_PB, mask_for_PB, centerOfBR, cv2.NORMAL_CLONE)
    # details about NORMAL_CLONE https://learnopencv2.com/seamless-cloning-using-opencv-python-cpp/
    # im_clone = cv2.cvtColor(im_clone, cv2.COLOR_BGR2RGB)
    im_clone = np.expand_dims(im_clone, axis=0)
    mixed = np.asarray(im_clone, dtype=np.uint8
    ) # 0~255

    Image.fromarray(mixed[0]).save(f'{outdir}/poisson_blend_{step_i}.png')
