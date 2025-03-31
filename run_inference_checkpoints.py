import os

input_dir = "/bdm-das/ADSP_v1/TestSet_Mindboggle-101/imagesTs"
base_output_dir = "/bdm-das/ADSP_v1/FastUNet_output/RTX_2080_Ti/DiceFocalLoss_noMirror"
os.makedirs(base_output_dir, exist_ok=True)
base_model_path = "/bdm-das/ADSP_v1/nnUNet_data/nnUNet_results/Dataset002_BrainStructure1K/" # nnUNetTrainerDiceTopK10Loss_NoMirroring__nnUNetPlans__3d_fullres_BN_4BS
trainer = "nnUNetTrainerDiceFocalLoss_NoMirroring" # Change Trainer (nnUNetTrainerNoMirroring, nnUNetTrainerDiceFocalLoss_NoMirroring, nnUNetTrainerTverskyLoss_NoMirroring)
plans = ['nnUNetPlans', 'nnUNetResEncUNetMPlans']
configs = ['3d_fullres_BN_4BS'] # 3d_fullres -> only for NoMirror trainer
checkpoints = [f"checkpoint_{i}" for i in range(100, 200, 100)] #+ ["checkpoint_final"]

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
