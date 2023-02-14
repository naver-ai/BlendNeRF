## BlendNeRF - Official PyTorch Implementation

<p align="middle"><img width="100%" src="assets/teaser.png" /></p>

> **3D-aware Blending with Generative NeRFs**<br>
> [Hyunsu Kim](https://blandocs.github.io)<sup>1</sup>, [Gayoung Lee](https://sites.google.com/site/gylee1103)<sup>1</sup>, [Yunjey Choi](https://yunjey.github.io)<sup>1</sup>, [Jin-Hwa Kim](http://wityworks.com)<sup>1,2</sup>, [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz)<sup>3</sup><br>
<sup>1</sup>NAVER AI Lab, <sup>2</sup>SNU AIIS, <sup>3</sup>CMU

[**Project page**](https://blandocs.github.io/blendnerf) | [**Arxiv**](https://arxiv.org/abs/2302.06608)

> **Abstract:** *Image blending aims to combine multiple images seamlessly. It remains challenging for existing 2D-based methods, especially when input images are misaligned due to differences in 3D camera poses and object shapes. To tackle these issues, we propose a 3D-aware blending method using generative Neural Radiance Fields (NeRF), including two key components: 3D-aware alignment and 3D-aware blending. For 3D-aware alignment, we first estimate the camera pose of the reference image with respect to generative NeRFs and then perform 3D local alignment for each part. To further leverage 3D information of the generative NeRF, we propose 3D-aware blending that directly blends images on the NeRF's latent representation space, rather than raw pixel space. Collectively, our method outperforms existing 2D baselines, as validated by extensive quantitative and qualitative evaluations with FFHQ and AFHQ-Cat.*

Code will come soon!



## Multi-view blending results
<hr>

<div>


### CelebA-HQ (EG3D / FFHQ-pretrained)
<p align="center">
<img src='assets/multiview_celeba.png' align="center" width=100%>
</p>

### AFHQv2-Cat (EG3D)
<p align="center">
<img src='assets/multiview_afhq.png' align="center" width=100%>
</p>

### ShapeNet-Car (EG3D)
<p align="center">
<img src='assets/multiview_shapenet.png' align="center" width=100%>
</p>

### FFHQ (StyleSDF)
<p align="center">
<img src='assets/multiview_stylesdf.png' align="center" width=100%>
</p>
</div>


## Comparison with baselines
<hr>
<div>

### CelebA-HQ
<p align="center">
<img src='assets/compare_celeba.png' align="center" width=100%>
</p>

### AFHQv2-Cat
<p align="center">
<img src='assets/compare_afhq.png' align="center" width=100%>
</p>
</div>
