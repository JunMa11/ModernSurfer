# FastUNet
Fast implementation for nnU-Net

# nnUNet Inference

## Overview
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) is the most popular package in medical image segmentation. 
Here we create a clean inference pipeline that only contains the necessary modules. 

## Installation

# Environment Setup

To ensure compatibility, use the following versions:

- **Python**: 3.10  
- **CUDA**: 12.4 

To install the nnUNet Inference module, you need to have latest pytorch installed on your system. Follow the steps below:

1. create env and clone the repo
```bash
conda create -n fast_unet python==3.10
conda activate fast_unet
git clone https://github.com/JunMa11/FastUNet.git
```

2. Install the packages:
```bash
pip install torch torchvision torchaudio
cd nnUNet
pip install -e .
pip install cupy-cuda12x
```
## Download dataset and model weights from the following link
https://drive.google.com/drive/folders/1WRu2v3Mr67mkf1lB_ZPyvRztvGu-htL8?usp=sharing

put the dataset in FastUnet/nnUNet_data and model weights folders in FastUnet/model_weights

## Running Inference

To run inference using a trained nnUNet model, follow these steps:

1. Use the following command to perform inference:

```bash
python nnunet_infer_nii.py -i <path_to_sample_data> -o <path_to_predictions> --model_path <path_to_model_weight>
```

For example

```bash
python nnunet_infer_nii.py -i /home/ys155/nnUNet_inference/sample_data/ -o ./seg --model_path /home/ys155/fastUNet/model_weights/701/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres
```


## (Optional) Running Inference with TensorRT.

Note: Current version of TensorRT will slow down the inference speed. InstanceNorm in TensorRT might be a problem.

Install the packages:

```bash
python -m pip install torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu124
pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com
cd ..
```

Inference with TensorRT:
```bash
python nnunet_infer_nii.py -i /home/ys155/nnUNet_inference/sample_data/ -o ./seg --model_path /home/ys155/fastUNet/model_weights/701/nnUNetTrainerMICCAI__nnUNetPlans__3d_fullres --trt
```