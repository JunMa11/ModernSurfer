# ModernSurfer
Accelerated version of nnU-Net for inference time

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
conda create -n modernsurfer python==3.10
conda activate modernsurfer
git clone https://github.com/JunMa11/ModernSurfer.git
```

2. Install the packages:
```bash
pip install torch torchvision torchaudio
cd ModernSurfer/nnUNet
pip install -e .
pip install cupy-cuda12x
```
## Download dataset (TestSet_Mindboggle-101) and the content of model weights(checkpoint_final.pth, dataset.json, and plans.json) from the following links
Dataset: https://drive.google.com/drive/folders/1PCMJdxUDFos9op9UurdosrAbOXWfjwQ2?usp=sharing
Model weights: https://drive.google.com/drive/folders/1ncHcP_xRi7xXhqpY7eSlssco88rA8vQ6?usp=sharing

put the dataset in ModernSurfer/nnUNet_data and model weights folders in ModernSurfer/model_weights

If you are familiar with nnUNet, the model weights are essentially obtained in the nnUNet_results folder. 
It requires the dataset.json and plans.json file to be outside of the "fold_all" folder while the checkpoint_final.pth should be inside the "fold_all" folder to make inference.

## Running Inference

To run inference using a trained nnUNet model, follow these steps:

1. Use the following command to perform inference:

```bash
python nnunet_infer_nii.py -i <path_to_sample_data> -o <path_to_predictions> --model_path <path_to_model_weight> --fold <fold_used_for_training> --checkpoint <checkpoint_for_prediction>
```

For example

```bash
python nnunet_infer_nii.py -i /home/user/ModernSurfer/nnUNet_data/TestSet_Mindboggle-101/imagesTs -o ./seg --model_path /home/user/ModernSurfer/model_weights/ --fold all --checkpoint checkpoint_final.pth
```

In the case that you want to run different models with different plans, configurations, or checkpoints, you can use the run_inference_checkpoints.py file.
In which you can change the input directory, output directory, model path, trainer, plans, configuration and checkpoints to make inferences for several different models. 
This file essentially run the same command above in sequence, it does NOT run the command in parallel with different GPU but rather run them in sequence on the same GPU.
