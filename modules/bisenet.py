"""
    Refer to https://github.com/CoinCheung/BiSeNet and https://github.com/zllrunning/face-parsing.PyTorch
"""


from itertools import chain
from pathlib import Path
from PIL import Image
from munch import Munch
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet18
from torchvision.utils import save_image

'''
| Label list | | |
| ------------ | ------------- | ------------ |
| 0: 'background' | 1: 'skin' | 2: 'r_eyebrow' |
| 3: 'l_eyebrow' | 4: 'r_eye' | 5: 'l_eye' |
| 6: 'glasses' | 7: 'r_ear' | 8: 'l_ear' |
| 9: 'earring' | 10: 'nose' | 11: 'inner_lip' |
| 12: 'upper_lip' | 13: 'lower_lip' | 14: 'neck' |
| 15: 'necklace' | 16: 'cloth' | 17: 'hair' |
| 18: 'hat' | | |
'''


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = resnet18()
        self.resnet._modules.pop('avgpool')
        self.resnet._modules.pop('fc')
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def resnet_(self, x):
        results = []
        for name, module in self.resnet._modules.items():
            x = module(x)
            if name in ['layer' + str(i) for i in [2, 3, 4]]:
                results.append(x)
            if name == 'layer4':
                break
        return results

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet_(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan//4, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(out_chan//4, out_chan, 1, 1, 0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    ''' It works with 512x512 image. Accuracy degrades in 256 or 1024. '''
    def __init__(self, fname_ckpt, n_classes=19, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.load_state_dict(torch.load(fname_ckpt))

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, _ = self.cp(x)  # here return res3b1 feature
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        return feat_out

    @torch.no_grad()
    def get_masks_19(self, x, intermediate_size=512):
        ''' input: [-1, 1] images of B x 3 x H x W
        outputs dict of 0-1 normalized masks '''
        scale_factor = x.size(2) / intermediate_size
        device = x.device
        x = F.interpolate(x, size=intermediate_size, mode='bilinear')
        x_01 = x*0.5 + 0.5
        imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
        x_imagenet = (x_01 - imagenet_mean) / imagenet_std
        parsed = self(x_imagenet)
        parsed = F.interpolate(parsed, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        masks = torch.zeros_like(parsed).to(device)
        parsed = parsed.argmax(1)
        for i in range(19):
            masks[:, i].masked_fill_(parsed == i, 1)
        return masks

    @torch.no_grad()
    def get_face_without_forehead(self, x, face_wing, intermediate_size=512):
        face_bisenet = self.get_masks(x).face
        return trim_forehead(face_bisenet, face_wing)

    @torch.no_grad()
    def get_masks(self, x, intermediate_size=512):
        masks = self.get_masks_19(x, intermediate_size)
        masks = Munch(face=(masks[:, 1:7].sum(1, keepdim=True) + masks[:, 9:14].sum(1, keepdim=True)), hair=masks[:, 17:19].sum(1, keepdim=True),
                      neck=masks[:, 14:15], clothes=masks[:, 16:17], bg=masks[:, :1], eyebrows=masks[:, 2:4].sum(1, keepdim=True), eyes=masks[:,4:7].sum(1, keepdim=True), nose=masks[:,10:11], lip=masks[:, 11:14].sum(1, keepdim=True),
                      parser='bisenet')
        return masks

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


@torch.no_grad()
def trim_forehead(face_bisenet, face_wing):
    B, _, H, W = face_bisenet.shape
    weight = torch.ones([B, H]).to(face_bisenet.device)
    for b in range(B):
        try:
            binary = (face_wing[b, 0] > 0.8).squeeze()
            start = min(torch.max(binary, dim=1)[0].nonzero())
            end = min(start + 10, H)
            weight[b, start:end] = torch.arange(0, 1, 1/(end-start).item())
            weight[b, :start] = 0
        except Exception as e:
            print(e)
            weight[b] = 1
    weight = weight.reshape(B, 1, H, 1)
    return face_bisenet * weight


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in 'png jpg jpeg JPG'.split(' ')]))
    return fnames


@torch.no_grad()
def test(dname_src, dname_dest, ckpt='bisenet.ckpt', image_size=512):
    from tqdm import tqdm
    os.makedirs(dname_dest, exist_ok=True)

    net = BiSeNet(ckpt).cuda().eval()

    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    images = []
    images_pil = []
    for fname in tqdm(listdir(dname_src), 'load images'):
        image = Image.open(fname).convert('RGB')
        image = image.resize((image_size, image_size), Image.BILINEAR)
        images_pil.append(image)
        images.append(to_tensor(image))

    images = torch.stack(images).cuda()
    outs = net(images)

    grid = []
    for image, out in tqdm(zip(images_pil, outs), 'visualize', len(outs)):
        parsing = out.cpu().numpy().argmax(0)
        vis_im = vis_parsing_maps(image, parsing)
        pair = [torch.Tensor(np.array(image)), torch.Tensor(vis_im)]
        grid.append(torch.cat(pair, dim=0))
    save_image(torch.stack(grid).permute(0, 3, 1, 2).float()/255, os.path.join(dname_dest, 'grid.jpg'), nrow=4)


def vis_parsing_maps(im, parsing_anno, stride=1):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    return vis_im


if __name__ == "__main__":
    from fire import Fire
    Fire(test)
