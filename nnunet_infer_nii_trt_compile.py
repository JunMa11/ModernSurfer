import numpy as np
import torch
import time
import os
import SimpleITK as sitk
import nnunetv2
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from typing import Tuple, Union
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from plain_unet import build_fix_plain_unet
from tqdm import tqdm
import argparse
import glob
import os



import torch.nn as nn
import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt as torchtrt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from modelopt.torch.quantization.utils import export_torch_mode


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                use_softmax,
                                                                return_probabilities: bool = False,
                                                                ):

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    # apply_inference_nonlin will convert to torch
    if use_softmax:
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)

        del predicted_logits
    
    # Start timing for converting probabilities to segmentation
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
    else:
        # Get the class with the maximum logit at each pixel
        max_logit, max_class = torch.max(predicted_logits, dim=0)
        
        # Apply threshold: Only assign the class if its logit exceeds the threshold
        segmentation = torch.where(max_logit >= 0.5, max_class, torch.tensor(0, device=predicted_logits.device))
    
    return segmentation

class SimplePredictor(nnUNetPredictor):
    """
    simple predictor for nnUNet
    """
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        print(trainer_class)
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')

        network = build_fix_plain_unet()

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        # network.load_state_dict(parameters[0])

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def preprocess(self, image, props):
        preprocessor = self.configuration_manager.preprocessor_class(verbose=False)
        data, _ = preprocessor.run_case_npy(image,
                                                  None,
                                                  props,
                                                  self.plans_manager,
                                                  self.configuration_manager,
                                                  self.dataset_json)
        data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        return data

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits
    

    def inference(self, image, properties_dict, use_softmax):
        image = self.preprocess(image, properties_dict)

        # for params in self.list_of_parameters:
        #     self.network.load_state_dict(params)

        with torch.no_grad():
            assert isinstance(image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()
            empty_cache(self.device)

            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                data, slicer_revert_padding = pad_nd_image(image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
                start_time = time.time()
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                            self.perform_everything_on_device)
                empty_cache(self.device) # Start time for inference time calculation
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
                segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits,
                                                                self.plans_manager,
                                                                self.configuration_manager,
                                                                self.label_manager,
                                                                properties_dict,
                                                                use_softmax,
                                                                return_probabilities=False,
                                                                )


                
        return segmentation.cpu().numpy()

def benchmark(model, input_shape=(1, 1, 96, 160, 160)):
    import torch
    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.xpu.is_available():
        device = 'xpu'
    else:
        print('Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices')
        import sys

        sys.exit(0)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    sort_by_keyword = device + "_time_total"

    data = torch.randn(input_shape).to("cuda")
    model = model.to("cuda")
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            out = model(data)
    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))



if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Inference for nnUNet model")
        parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input image file')
        parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output segmentation')
        parser.add_argument('--model_path', type=str, required=True, help='Name of the model to use for inference')
        parser.add_argument('--fold', type=str, default='all', help='Fold number to use for inference (default: 0)')
        parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pth', help='Path to the model checkpoint file')
        parser.add_argument('--use_softmax', default=True, help='Apply softmax to the output probabilities')
        return parser.parse_args()

    args = parse_arguments()

    predictor = SimplePredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    predictor.initialize_from_trained_model_folder(
        args.model_path,
        use_folds= args.fold,
        checkpoint_name= args.checkpoint,
    )

    input_folder = args.input_path
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, '*'))

    input_shape = (1, 1, 96, 160, 160)
    model = predictor.network
    model.cuda()
    model.eval()

    data = torch.randn(input_shape).to("cuda")
    for _ in range(10):
        features = model(data)

    torch.cuda.synchronize()
    for i in tqdm(range(500)):
        out = model(data)
        torch.cuda.synchronize()

    model.eval()
    benchmark(model)

    with torch.no_grad():
        with export_torch_mode():
            # Compile the model with Torch-TensorRT Dynamo backend
            # input_tensor = images.cuda()
            input_tensor = torch.randn(input_shape).to("cuda")
            # torch.export.export() failed due to RuntimeError: Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()
            from torch.export._trace import _export

            exp_program = _export(model, (input_tensor,))
            # exp_program = torchtrt.dynamo.export(model, (input_tensor,))
            # enabled_precisions = {torch.float}
            enabled_precisions = {torch.half} # , torch.int8 torch.half,
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                enabled_precisions=enabled_precisions,
                min_block_size=1,
            )

            data = torch.randn(input_shape).to("cuda")

            for _ in tqdm(range(10)):
                features = trt_model(data)

            torch.cuda.synchronize()
            for i in tqdm(range(500)):
                out = trt_model(data)
                torch.cuda.synchronize()

            benchmark(trt_model)

            predictor.network = trt_model

            for file in tqdm(files):
                image, props = img, props = SimpleITKIO().read_images([file])
                seg = predictor.inference(image, props, args.use_softmax)
                sitk_img = sitk.GetImageFromArray(seg)
                case_name = file.split('/')[-1].replace('_0000.nii.gz', '.nii.gz')
                # case_name = file.split('/')[-1].replace('.nii.gz', '.nii.gz')
                sitk.WriteImage(sitk_img, os.path.join(output_folder, f'{case_name}'))





