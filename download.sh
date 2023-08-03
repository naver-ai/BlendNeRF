NETWORK_FOLDER="./checkpoints"
mkdir -p $NETWORK_FOLDER

# EG3D generators
# https://github.com/NVlabs/eg3d
# afhqcats512-128.pkl: same as in the EG3D repo
# original_ffhq_512-128_400kimg.pkl: finetuned from ffhq512-128.pkl in the EG3D repo

# Segmentation Networks
# Human Face: https://github.com/zllrunning/face-parsing.PyTorch
# Animal Face: https://github.com/nv-tlabs/datasetGAN_release

URLs=(
    1D8zhelySaUDVYpDmdXh6hdz0QI4lg6wl
    1SQALnkETqVB4xWyt_48ZR7Vs1tNEDq2i
    1qWUo3moSWytw1rBoTYuaJU2yqstinKmb
    1DakZ64nm-UtXwp10OYz7qOJyiTDxtLK2
    1HsY6fFMZ5UnOi8uhXOwypjc5pRxbDMlx
    15NGZFILjkKePyuON3TBNo92TDFee6a-m
)

FILENAMEs=(
    afhqcats512-128.pkl
    original_ffhq_512-128_400kimg.pkl
    encoder_afhq.pkl
    encoder_ffhq.pkl
    bisenet.ckpt
    deeplab_epoch_19.pth
)

len=${#URLs[@]}
for (( idx = 0; idx < len; idx++ ));
do
    gdown "https://docs.google.com/uc?export=download&id=${URLs[idx]}"
    mv ${FILENAMEs[idx]} $NETWORK_FOLDER/
    echo "https://docs.google.com/uc?export=download&id=${URLs[idx]}"
done