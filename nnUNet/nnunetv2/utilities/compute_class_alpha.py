import os
import numpy as np
import torch
import nibabel as nib
from multiprocessing import Pool, cpu_count
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


def _count_voxels_worker(args):
    seg_path, num_classes, ignore_label = args
    seg = nib.load(seg_path).get_fdata().astype(np.int64)
    flat = seg.flatten()

    counts = np.zeros(num_classes, dtype=np.int64)
    for class_id in np.unique(flat):
        if ignore_label is not None and class_id == ignore_label:
            continue
        if class_id < num_classes:
            counts[class_id] += np.sum(flat == class_id)
    return counts


def compute_class_alpha(preprocessed_dataset_folder_base: str, label_manager, ignore_label: int = None) -> torch.Tensor:
    num_classes = label_manager.num_segmentation_heads
    label_counts = np.zeros(num_classes, dtype=np.int64)

    label_folder = os.path.join(preprocessed_dataset_folder_base, 'gt_segmentations')
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.nii.gz')]
    full_paths = [os.path.join(label_folder, f) for f in label_files]

    # Prepare args for parallel processing
    args = [(path, num_classes, ignore_label) for path in full_paths]

    # Use your cluster-aware CPU allocation function
    num_workers = get_allowed_n_proc_DA()

    with Pool(processes=num_workers) as pool:
        results = pool.map(_count_voxels_worker, args)

    for counts in results:
        label_counts += counts

    label_counts = np.maximum(label_counts, 1)
    inv_freq = 1.0 / label_counts
    alpha = inv_freq / inv_freq.sum()

    return torch.tensor(alpha, dtype=torch.float32)

