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
