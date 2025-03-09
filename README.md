# CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution [CVPR2025]
This repository is an official implementation of the paper "CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution", CVPR, 2025.

## Contents
1. [Enviroment](#Environment)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Acknowledgements](#Acknowledgements)


## Environment
- Python 3.9
- PyTorch >=2.2

### Installation
```bash
pip install -r requirements.txt
python setup.py develop
```




## Training
### Data Preparation
- Download the training dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and put them in the folder `./datasets/SR`.
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets/SR`.
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




## Testing
### Data Preparation
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets`.

### Pretrained Models
- Modify the paths to the pretrained model files in the test files of the `./options/test` folder to ensure they point to the correct pretrained model files.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.


```bash
python basicsr/test.py -opt options/test/test_CATANet_x2.yml
python basicsr/test.py -opt options/test/test_CATANet_x3.yml
python basicsr/test.py -opt options/test/test_CATANet_x4.yml
```




## Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
