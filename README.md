# FastUNet
Fast implementation for nnU-Net

# nnUNet Inference

## Overview
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) is the most popular package in medical image segmentation. 
Here we create a clean inference pipeline that only contains the necessary modules. 

## Installation

To install the nnUNet Inference module, you need to have latest pytorch installed on your system. Follow the steps below:

1. Clone the repository:
```bash
cd nnUNet
pip install -e .
```


## Running Inference

To run inference using a trained nnUNet model, follow these steps:

1. Use the following command to perform inference:

```bash
python nnunet_infer_nii.py -i <path_to_sample_data> -o <path_to_predictions> --model_path <path_to_model_weight/nnUNetTrainerDA5__nnUNetPlans__3d_lowres>
```

For example

```bash
python nnunet_infer_nii.py -i sample_data/ -o ./seg --model_path model_weight/nnUNetTrainerDA5__nnUNetPlans__3d_lowres/
```

## Accelerating nnUNet-fp16 using TensorRT. 
Note:
1. Line 17 makes the nnUNet use BatchNorm3D.
2. Line 121 in nnunet_infer_nii_trt_compile.py, "network.load_state_dict(parameters[0])" is disabled since I changed the InstanceNorm into BatchNorm and did not load the pre-trained model. 

### Check the log.txt for the inference time.
```bash
conda create -n nnunet python=3.10
conda activate nnunet
cd nnUNet
pip install -e .
python -m pip install torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu124
pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com
cd ..

CUDA_VISIBLE_DEVICES=0 python nnunet_infer_nii_trt_compile.py -i sample_data/ -o ./seg_trt --model_path model_weight/nnUNetTrainerDA5__nnUNetPlans__3d_lowres/ > log.txt 2>&1 
```

FYI My Version:
```bash
torch.__version__ '2.6.0+cu124'
torch_tensorrt.__version__ '2.6.0+cu124'
```
Install Cupy for preprocess function
```bash
pip install cupy-cuda12x
```
