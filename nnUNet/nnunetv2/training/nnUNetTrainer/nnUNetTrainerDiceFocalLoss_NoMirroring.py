from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_focal_loss import DC_and_Focal_Loss
from nnunetv2.utilities.compute_class_alpha import compute_class_alpha
import numpy as np

class nnUNetTrainerDiceFocalLoss_NoMirroring(nnUNetTrainer):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    
    def _build_loss(self):
        alpha = compute_class_alpha(self.preprocessed_dataset_folder_base, self.label_manager, self.label_manager.ignore_label)
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = DC_and_Focal_Loss(
            {'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            alpha=alpha,
            gamma=2.0,
            ignore_index=self.label_manager.ignore_label,
        )
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss