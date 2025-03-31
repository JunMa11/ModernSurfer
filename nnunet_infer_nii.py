import numpy as np
import torch
from time import time
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
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from nnunetv2.architecture.repvgg_unet import plain_unet_S5, plain_unet_S4, plain_unet_702, plain_unet
from nnunetv2.preprocessing.resampling.default_resampling import fast_resample_logit_to_shape
from tqdm import tqdm
import argparse
import glob
import os
import gc
from collections import OrderedDict
import pandas as pd
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
    if properties_dict['shape_after_cropping_and_before_resampling'][0] < 600:
        t0 = time()
        predicted_logits = fast_resample_logit_to_shape(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
        postprocess_resample_time = time() - t0

        gc.collect()
        empty_cache(predicted_logits.device)
        t0_logit_to_seg = time()
        if use_softmax:
            predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)

            del predicted_logits
            
            # Start timing for converting probabilities to segmentation
            segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

        else:
            t0_logit_to_seg = time()
            # Get the class with the maximum logit at each pixel
            max_logit, max_class = torch.max(predicted_logits, dim=0)
                
                # Apply threshold: Only assign the class if its logit exceeds the threshold
            segmentation = torch.where(max_logit >= 0.5, max_class, torch.tensor(0, device=predicted_logits.device))
        logit_to_seg_time = time() - t0_logit_to_seg
    else:
        t0 = time()
        segmentation = fast_resample_logit_to_shape(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
        postprocess_resample_time = time() - t0



    dtype = torch.uint8 if len(label_manager.foreground_labels) < 255 else torch.uint16
    segmentation_reverted_cropping = torch.zeros(properties_dict['shape_before_cropping'], dtype=dtype)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation

    del segmentation

    # Revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.permute(plans_manager.transpose_backward)

    return segmentation_reverted_cropping.cpu(), logit_to_seg_time, postprocess_resample_time

class SimplePredictor(nnUNetPredictor):
    """
    simple predictor for nnUNet
    """
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str):
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
            ckpt = checkpoint['network_weights']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            parameters.append(ckpt)

        configuration_manager = plans_manager.get_configuration(configuration_name)
        print(f"This is the trainer name: {trainer_name}")
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        
        print(f'Using trainer class: {trainer_class}')

        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        if 'S4' in model_training_output_dir:
            network = plain_unet_S4(14, False, False)
        elif 'S5' in model_training_output_dir:
            network = plain_unet_S5(14, False, False)
        else:
            network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
            )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])
        for params in self.list_of_parameters:
            self.network.load_state_dict(params)
        
        for module in self.network.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

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
        image = torch.from_numpy(image).to(dtype=torch.float32, memory_format=torch.contiguous_format).to(self.device)
        data, resample_time = preprocessor.run_case_npy(image,
                                                  None,
                                                  props,
                                                  self.plans_manager,
                                                  self.configuration_manager,
                                                  self.dataset_json)
        #data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        return data, resample_time

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
        image, resample_time = self.preprocess(image, properties_dict)


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

                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                            self.perform_everything_on_device)
                print(f'predicted_logits shape: {predicted_logits.shape}')
                empty_cache(self.device) # Start time for inference time calculation
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

                segmentation, logit_to_seg_time, postprocess_resample_time = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits,        
                                                                self.plans_manager,
                                                                self.configuration_manager,
                                                                self.label_manager,
                                                                properties_dict,
                                                                use_softmax,
                                                                return_probabilities=False,
                                                                )
                print(f'segmentation shape: {segmentation.shape}')
                print(f'Logit to seg time:: {logit_to_seg_time:.4f}')
                print(f'Postprocessing resampling time: {postprocess_resample_time:.6f}')

        return segmentation,resample_time, logit_to_seg_time, postprocess_resample_time

if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Inference for nnUNet model")
        parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input image file')
        parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output segmentation')
        parser.add_argument('--model_path', type=str, required=True, help='Name of the model to use for inference')
        parser.add_argument('--fold', type=str, default='all', help='Fold number to use for inference (default: 0)')
        parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth', help='Path to the model checkpoint file')
        parser.add_argument('--use_softmax', default=False, help='Apply softmax to the output probabilities')
        parser.add_argument('--trt', action='store_true', help='Using TensorRT')
        parser.add_argument('--onnx_trt', action='store_true', help='Using TensorRT')
        parser.add_argument('--run_engine_trt', action='store_true', help='Using TensorRT')

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

    prediction_results = []

    if args.onnx_trt:
        model = predictor.network
        model.cuda()
        model.eval()
        input_shape = (1, 1, 64, 256, 256)
        # prediction.shape (14, 64, 256, 256)
        input_tensor = torch.randn(input_shape).to("cuda")

        # onnx_program = torch.onnx.dynamo_export(model, input_tensor)
        # if "onnx_models" not in os.listdir():
        #     os.makedirs("onnx_models")
        # onnx_program.save("onnx_models/fast_unet.onnx")

        onnx_file_path = "onnx_models/fast_unet_fp32.onnx"
        if "onnx_models" not in os.listdir():
            os.makedirs("onnx_models")
        torch.onnx.export(
            model,
            input_tensor,
            onnx_file_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=16,
            export_params=True,
            keep_initializers_as_inputs=True,
        )

        print(f"ONNX model exported to {onnx_file_path}")

        import numpy as np

        # Example numpy file for single-input ONNX
        calib_data = np.random.randn(1, 1, 64, 256, 256)
        calib_data = calib_data.astype(np.float32)
        np.save("calib_data.npy", calib_data)

        # Example numpy file for single/multi-input ONNX
        # Dict key should match the input names of ONNX
        # calib_data = {
        #     "input_name": np.random.randn(*shape),
        #     "input_name2": np.random.randn(*shape2),
        # }
        # np.savez("/workspace/calib_data.npz", calib_data)

        import sys

        sys.path.insert(0, './TensorRT-Model-Optimizer')

        import modelopt.onnx.quantization as moq
        import numpy as np

        calibration_data_path = 'calib_data.npy'
        # onnx_path = "vit_base_patch16_224.onnx"
        #
        calibration_data = np.load(calibration_data_path)

        # moq.quantize(
        #     onnx_path="onnx_models/fast_unet.onnx",
        #     calibration_data=calibration_data,
        #     output_path="onnx_models/quant_fast_unet.onnx",
        #     quantize_mode="int8",
        #     # high_precision_dtype="float16",
        # )

        moq.quantize(
            onnx_path="onnx_models/fast_unet_fp32.onnx",
            calibration_data=calibration_data,
            calibration_method='max',
            output_path="onnx_models/quant_fast_unet_int8.onnx",
            quantize_mode="int8",
            high_precision_dtype="float32",
        )

        # ! / usr / src / tensorrt / bin / trtexec - -onnx = onnx_models / quant_fast_unet.onnx - -saveEngine = onnx_models / quant_fast_unet.engine - -best
        # ! / usr / src / tensorrt / bin / trtexec - -onnx = onnx_models / quant_fast_unet.onnx - -saveEngine = onnx_models / quant_fast_unet_fp16.engine - -fp16

    if args.run_engine_trt:
        import sys

        sys.path.insert(0, './TensorRT-Model-Optimizer')
        from modelopt.torch._deploy._runtime import RuntimeRegistry
        from modelopt.torch._deploy.device_model import DeviceModel
        from modelopt.torch._deploy.utils import OnnxBytes

        # Configure deployment
        deployment = {
            "runtime": "TRT",
            "version": "10.3",
            # "precision": 'int8',
            "precision": 'fp32',
        }


        # Create an ONNX bytes object
        onnx_bytes = OnnxBytes('onnx_models/quant_fast_unet_int8.onnx').to_bytes()

        # Get the runtime client
        client = RuntimeRegistry.get(deployment)

        # Compile the TRT model
        print("Compiling the TensorRT engine. This may take a few minutes...")
        compiled_model = client.ir_to_compiled(onnx_bytes)
        print("Compilation completed.")

        # Print size of the compiled model
        engine_size = len(compiled_model)
        print(f"Size of the TensorRT engine: {engine_size / (1024 ** 2):.2f} MB")

        # Create the device model
        device_model = DeviceModel(client, compiled_model, metadata={})
        print(f"Inference latency reported by device_model: {device_model.get_latency()} ms")

        # device_model.eval()
        predictor.network = device_model

        for file in tqdm(files):
            image, props = SimpleITKIO().read_images([file])
            t0 = time()
            seg, resample_time, logit_to_seg_time, postprocess_resample_time = predictor.inference(image, props, args.use_softmax)
            elapsed_time = time() - t0
            print(f'total: {time() - t0}')
            sitk_img = sitk.GetImageFromArray(seg)
            sitk_img.SetSpacing(props['sitk_stuff']['spacing'])
            sitk_img.SetOrigin(props['sitk_stuff']['origin'])
            sitk_img.SetDirection(props['sitk_stuff']['direction'])
            case_name = file.split('/')[-1].replace('_0000.nii.gz', '.nii.gz')
            sitk.WriteImage(sitk_img, os.path.join(output_folder, f'{case_name}'))

            result = OrderedDict()
            result["Filename"] = case_name
            result["Prediction Time (seconds)"] = elapsed_time
            result["Preprocess Resampling time (seconds)"] = resample_time
            result["Logit to segmentation time (seconds)"] = logit_to_seg_time
            result["Postprocessing Resampling time (seconds)"] = postprocess_resample_time
            prediction_results.append(result)

        exit(0)

    if args.trt:
        import torch_tensorrt as torchtrt
        # it becomes slow in this version
        model = predictor.network
        model.cuda()
        model.eval()
        input_shape = (1, 1, 64, 256, 256)
        with torch.no_grad():
            with export_torch_mode():
                input_tensor = torch.randn(input_shape).to("cuda")
                from torch.export._trace import _export

                exp_program = _export(model, (input_tensor,))
                enabled_precisions = {torch.half}
                trt_model = torchtrt.dynamo.compile(
                    exp_program,
                    inputs=[input_tensor],
                    enabled_precisions=enabled_precisions,
                    min_block_size=1,
                )
                predictor.network = trt_model
            for file in tqdm(files):
                image, props = SimpleITKIO().read_images([file])
                t0 = time()
                seg, resample_time, logit_to_seg_time, postprocess_resample_time = predictor.inference(image, props, args.use_softmax)
                elapsed_time = time() - t0
                print(f'total prediction time: {time() - t0}')
                sitk_img = sitk.GetImageFromArray(seg)
                sitk_img.SetSpacing(props['sitk_stuff']['spacing'])
                sitk_img.SetOrigin(props['sitk_stuff']['origin'])
                sitk_img.SetDirection(props['sitk_stuff']['direction'])
                case_name = file.split('/')[-1].replace('_0000.nii.gz', '.nii.gz')
                sitk.WriteImage(sitk_img, os.path.join(output_folder, f'{case_name}'))

                result = OrderedDict()
                result["Filename"] = case_name
                result["Prediction Time (seconds)"] = elapsed_time
                result["Preprocess Resampling time (seconds)"] = resample_time
                result["Logit to segmentation time (seconds)"] = logit_to_seg_time
                result["Postprocessing Resampling time (seconds)"] = postprocess_resample_time
                prediction_results.append(result)
    else:
        for file in tqdm(files):
            image, props = SimpleITKIO().read_images([file])
            t0 = time()
            seg, resample_time, logit_to_seg_time, postprocess_resample_time = predictor.inference(image, props, args.use_softmax)
            elapsed_time = time() - t0
            print(f'total: {time() - t0:.4f}')
            sitk_img = sitk.GetImageFromArray(seg)
            sitk_img.SetSpacing(props['sitk_stuff']['spacing'])
            sitk_img.SetOrigin(props['sitk_stuff']['origin'])
            sitk_img.SetDirection(props['sitk_stuff']['direction'])
            case_name = file.split('/')[-1].replace('_0000.nii.gz', '.nii.gz')
            sitk.WriteImage(sitk_img, os.path.join(output_folder, f'{case_name}'))
            
            result = OrderedDict()
            result["Filename"] = case_name
            result["Prediction Time (seconds)"] = elapsed_time
            result["Preprocess Resampling time (seconds)"] = resample_time
            result["Logit to segmentation time (seconds)"] = logit_to_seg_time
            result["Postprocessing Resampling time (seconds)"] = postprocess_resample_time
            prediction_results.append(result)

    # Convert list of OrderedDicts to DataFrame
    results_df = pd.DataFrame(prediction_results)

    # Save results to an Excel file
    results_df.to_excel(output_folder+"/inference_times.xlsx", index=False, engine="openpyxl")

    print(f"Prediction times saved to {output_folder}")




