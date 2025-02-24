import SimpleITK as sitk
import os

# Define the directory containing the .nii.gz files
nii_folder = '/home/jma/Documents/Ching-Yuan/miccai/nnUNet_data/validation/Dataset702_AbdomenMR/imagesTs'  # Update this path accordingly

# Iterate through all .nii.gz files in the specified directory
for filename in os.listdir(nii_folder):
    if filename.endswith('.nii.gz'):
        file_path = os.path.join(nii_folder, filename)
        img = sitk.ReadImage(file_path)
        print(f"{filename}: {img.GetSize()}")

