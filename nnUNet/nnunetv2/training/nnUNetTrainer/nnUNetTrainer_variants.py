from typing import List, Union, Tuple

import numpy as np
import torch
import copy
from torch import autocast, nn
from torch import distributed as dist
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessTransform, ContrastAugmentationTransform, \
    GammaTransform
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalGammaTransform
from batchgenerators.transforms.noise_transforms import MedianFilterTransform, GaussianBlurTransform, \
    GaussianNoiseTransform, BlankRectangleTransform, SharpeningTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, Rot90Transform, TransposeAxesTransform, \
    MirrorTransform
from batchgenerators.transforms.utility_transforms import OneOfTransform, RemoveLabelTransform, RenameTransform, \
    NumpyToTensor
from batchgeneratorsv2.helpers.scalar_type import RandomScalar

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert3DTo2DTransform, \
    Convert2DTo3DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.loss.robust_ce_loss import TopKLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from time import time
from nnunetv2.utilities.collate_outputs import collate_outputs
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.architecture.repvgg_unet import plain_unet, plain_unet_S4, plain_unet_S5, plain_unet_702


class nnUNetTrainerMICCAI(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))


        if (current_epoch + 1) % 100 == 0:
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch+1}.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))


        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1


class nnUNetTrainerMICCAI_repvgg(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        

        return plain_unet(num_output_channels, None, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg_l5(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        convs_per_stage = ['stackedconv', 'stackedconv', 'repvgg', 'repvgg', 'repvgg', 'repvgg', 'repvgg']
        

        return plain_unet(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg_l4(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        convs_per_stage = ['stackedconv', 'stackedconv', 'stackedconv', 'repvgg', 'repvgg', 'repvgg', 'repvgg']
        

        return plain_unet(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg_l3(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        convs_per_stage = ['stackedconv', 'stackedconv', 'stackedconv', 'stackedconv', 'repvgg', 'repvgg', 'repvgg']
        

        return plain_unet(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg_f3(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        convs_per_stage = ['repvgg', 'repvgg', 'repvgg', 'stackedconv', 'stackedconv', 'stackedconv', 'stackedconv']
        

        return plain_unet(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg_f4(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        convs_per_stage = ['repvgg', 'repvgg', 'repvgg', 'repvgg', 'stackedconv', 'stackedconv', 'stackedconv']
        

        return plain_unet(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg_f5(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        convs_per_stage = ['repvgg', 'repvgg', 'repvgg', 'repvgg', 'repvgg', 'stackedconv', 'stackedconv']
        

        return plain_unet(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg702(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        

        return plain_unet_702(num_output_channels, None, enable_deep_supervision, False)


class nnUNetTrainerMICCAI_repvgg702_l2(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        convs_per_stage = ['stackedconv', 'stackedconv', 'stackedconv', 'stackedconv', 'repvgg', 'repvgg']

        return plain_unet_702(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg702_l3(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        convs_per_stage = ['stackedconv', 'stackedconv', 'stackedconv', 'repvgg', 'repvgg', 'repvgg']

        return plain_unet_702(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg702_l4(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        convs_per_stage = ['stackedconv', 'stackedconv', 'repvgg', 'repvgg', 'repvgg', 'repvgg']

        return plain_unet_702(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg702_f2(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        convs_per_stage = ['repvgg', 'repvgg', 'stackedconv', 'stackedconv', 'stackedconv', 'stackedconv']

        return plain_unet_702(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg702_f3(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        convs_per_stage = ['repvgg', 'repvgg', 'repvgg', 'stackedconv', 'stackedconv', 'stackedconv']

        return plain_unet_702(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg702_f4(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        convs_per_stage = ['repvgg', 'repvgg', 'repvgg', 'repvgg', 'stackedconv', 'stackedconv']

        return plain_unet_702(num_output_channels, convs_per_stage, enable_deep_supervision, False)

class nnUNetTrainerMICCAI_repvgg_S4(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        model = plain_unet_S4(num_output_channels, True, False)
        return model

class nnUNetTrainerMICCAI_repvgg_S5(nnUNetTrainerMICCAI):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        model = plain_unet_S5(num_output_channels, True, False)
        return model
    