import os

input_dir = "/data_input/TestSet_Mindboggle-101/imagesTs"   #Change this to the path of the dataset
base_output_dir = "/data_output/TestSet_Mindboggle-101/predTs" #Change this to where you want the segmentation to output
os.makedirs(base_output_dir, exist_ok=True)
base_model_path = "/model_weight/checkpoint_final.pth" # Change this to your model path, the .pth file!
trainer = "nnUNetTrainerNoMirroring" # Change the trainer
plans = ['nnUNetPlans', 'nnUNetResEncUNetMPlans'] # This is the current plans that are availabe in nnUNet
configs = ['3d_fullres_BN_4BS'] # 3d_fullres -> only for NoMirror trainer
checkpoints = [f"checkpoint_{i}" for i in range(100, 1000, 100)] + ["checkpoint_final"] # This can be changed to the checkpoints you do have

for plan in plans:
    for config in configs:
        for checkpoint in checkpoints:
            checkpoint_dir = os.path.join(base_output_dir, checkpoint)
            os.makedirs(checkpoint_dir, exist_ok=True)
            output_dir = os.path.join(checkpoint_dir, f"{plan}_{config}")
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(base_model_path, f"{trainer}__{plan}__{config}")
            checkpoint_file = f"{checkpoint}.pth"
            print(f"Trainer: {trainer}  Plan: {plan}   Config: {config}    Checkpoint: {checkpoint}")
            cmd = f"python nnunet_infer_nii.py -i {input_dir} -o {output_dir} --model_path {model_path} --fold all --checkpoint {checkpoint_file}"
            print(f"Running: {cmd}")
            os.system(cmd)
