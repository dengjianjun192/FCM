## Semi-supervised medical image segmentation via feature similarity and reliable-region enhancement

**Authors**: Youjia Fu, Jianjun Deng, Shengkai Hu, and Antao Lin

### News
```
 <02.24.2025> We released the codes
```
---

### Introduction
Official code for "Semi-supervised medical image segmentation via feature similarity and reliable-region enhancement."
### Requirements
This repository is based on PyTorch 2.0.0, CUDA 11.8 and Python 3.8. All experiments in our paper were conducted on a single NVIDIA V100 GPU with an identical experimental setting.

### Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/dengjianjun192/FCM.git
 2. Put the data in './FCM/project/data'
 3. Train the model:
    ```
    python train.py  #for ACDC training
    ```
2. test the model
     ```
    python test_2D_ACDC.py
   ```
### Acknowledgements
Our code is adapted from [MC-Net](https://github.com/ycwu1997/MC-Net), [SSNet](https://github.com/ycwu1997/SS-Net), and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks to these authors for their valuable works and hope our model can promote the relevant research as well.
