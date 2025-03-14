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
```

Inference with TensorRT:
```bash
python nnunet_infer_nii.py -i sample_data/ -o ./seg --model_path model_weights/701/nnUNetTrainerMICCAI__nnUNetPlans__3d_fullres --trt
```

## (Optional) Running Inference with Onnx TensorRT.
```bash
pip install onnx
pip install onnxscript
pip install timm optimum

Download TensorRT:
https://developer.nvidia.com/tensorrt/download/10x
Install TensorRT:
https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#

install cudnn9
sudo apt-get -y install cudnn9-cuda-12

add 

quantizable_op_types = [op for op in quantizable_op_types if op != "ConvTranspose"]
in line 150 in /Users/yangsui/Documents/Research/FastUNet/TensorRT-Model-Optimizer/modelopt/onnx/quantization/int8.py,
to not to quantize ConvTranspose. Otherwise, Error: assert not np_y_scale.shape or w32.shape[-1] == np_y_scale.shape[0].
```
Run:
```bash
python nnunet_infer_nii.py -i sample_data/ -o ./seg --model_path model_weights/701/nnUNetTrainerMICCAI__nnUNetPlans__3d_fullres --onnx_trt 
```

But python run_engien_unsuccessful.py cannot be run. Need to check.


New Mar 8 2025
```bash
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
tar -xvzf TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz

pip install -r requirements.txt

python nnunet_infer_nii.py -i sample_data/ -o ./seg --model_path model_weights/701/nnUNetTrainerMICCAI__nnUNetPlans__3d_fullres --onnx_trt

#python run_engine_trt.py

python nnunet_infer_nii.py -i /home/ys155/nnUNet_inference/sample_data/ -o ./seg --model_path /home/ys155/fastUNet/model_weights/701/nnUNetTrainerMICCAI__nnUNetPlans__3d_fullres --run_engine_trt

trtexec --onnx=/tmp/tmpwumrwl2f/onnx/quant_fast_unet_int8.onnx --fp16 --int8 --saveEngine=/tmp/tmpwumrwl2f/quant_fast_unet_int8/quant_fast_unet_int8.engine --skipInference --builderOptimizationLevel=4 --verbose --exportLayerInfo=/tmp/tmpwumrwl2f/quant_fast_unet_int8/quant_fast_unet_int8.engine.graph.json

```


Mar 13 2025; only last step fails: Error: version is not matched.
```bash

git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
python -m pip install torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu124
pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com

# transform the pytorch model into onnx file; transform fp32 onnx into INT8 onnx.
python nnunet_infer_nii.py -i sample_data/ -o ./seg --model_path model_weights/701/nnUNetTrainerMICCAI__nnUNetPlans__3d_fullres --onnx_trt

# install TensorRT 10.3
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
tar -xvzf TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz

TENSORRT_PATH=$(pwd)/TensorRT-10.3.0.26
echo "export TENSORRT_HOME=${TENSORRT_PATH}" >> ~/.bashrc
echo "export PATH=\$TENSORRT_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
ln -sf /workspace/TensorRT-10.3.0.26/bin/trtexec /usr/local/bin/trtexec
source ~/.bashrc

sudo apt-get update
sudo apt-get -y install tensorrt
pip install -r requirements.txt
pip install "numpy<2.0.0"
pip install torch torchvision --upgrade

# running with the trt engine
python nnunet_infer_nii.py -i sample_data/ -o ./seg --model_path model_weights/701/nnUNetTrainerMICCAI__nnUNetPlans__3d_fullres --run_engine_trt

```

