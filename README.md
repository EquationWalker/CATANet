# CATANet -- CVPR2025
This repository is an official implementation of the paper "CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution", CVPR, 2025. 

### [[arXiv](https://arxiv.org/abs/2503.06896)] 

## :newspaper:News

- :white_check_mark: 2025-03-12:  arXiv paper available.
- :white_check_mark:2025-03-09: Release the codes of our CATANet.

> **Abstract:**   Transformer-based methods have demonstrated impressive performance in low-level visual tasks such as Image Super-Resolution (SR). However, its computational complexity grows quadratically with the spatial resolution. A series of works attempt to alleviate this problem by dividing Low-Resolution images into local windows, axial stripes, or dilated windows. SR typically leverages the redundancy of images for reconstruction, and this redundancy appears not only in local regions but also in long-range regions. However, these methods limit attention computation to content-agnostic local regions, limiting directly the ability of attention to capture long-range dependency. To address these issues, we propose a lightweight Content-Aware Token Aggregation Network (CATANet). Specifically, we propose an efficient Content-Aware Token Aggregation module for aggregating long-range content-similar tokens, which shares token centers across all image tokens and updates them only during the training phase. Then we utilize intra-group self-attention to enable long-range information interaction. Moreover, we design an inter-group cross-attention to further enhance global information interaction. The experimental results show that, compared with the state-of-the-art cluster-based method SPIN, our method achieves superior performance, with a maximum PSNR improvement of $\textbf{\textit{0.33dB}}$ and nearly $\textbf{\textit{double}}$ the inference speed.

⭐If this work is helpful for you, please help star this repo. Thanks!🤗

## :bookmark_tabs:Contents
1. [Enviroment](#Environment)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Citation](#Citation)
1. [Contact](#Contact)
1. [Acknowledgements](#Acknowledgements)


## :hammer:Environment
- Python 3.9
- PyTorch >=2.2

### Installation
```bash
pip install -r requirements.txt
python setup.py develop
```




## :rocket:Training
### Data Preparation
- Download the training dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and put them in the folder `./datasets`.
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[Download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets`.
- It's recommanded to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.

### Training Commands
- Refer to the training configuration files in `./options/train` folder for detailed settings.
```bash
# batch size = 4 (GPUs) × 16 (per GPU)
# training dataset: DIV2K

# ×2 scratch, input size = 64×64,800k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 basicsr/train.py -opt options/train/train_CATANet_x2_scratch.yml

# ×3 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 basicsr/train.py -opt options/train/train_CATANet_x3_finetune.yml

# ×4 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 basicsr/train.py -opt options/train/train_CATANet_x4_finetune.yml
```




## :wrench:Testing
### Data Preparation
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[Download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets`.

### Pretrained Models
- Download the [pretrained models] and put them in the folder `./pretrained_models`.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.


```bash
python basicsr/test.py -opt options/test/test_CATANet_x2.yml
python basicsr/test.py -opt options/test/test_CATANet_x3.yml
python basicsr/test.py -opt options/test/test_CATANet_x4.yml
```

## 🥰Citation

Please cite us if our work is useful for your research.

```
@article{liu2025CATANet,
  title={CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution},
  author={Xin Liu and Jie Liu and Jie Tang and Gangshan Wu},
  journal={arXiv preprint arXiv:2503.06896},
  year={2025}
}
```

## :mailbox:Contact

If you have any questions, feel free to approach me at xinliu2023@smail.nju.edu.cn

## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
