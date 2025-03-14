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

moq.quantize(
    onnx_path="onnx_models/fast_unet_fp32.onnx",
    calibration_data=calibration_data,
    calibration_method='max',
    output_path="onnx_models/quant_fast_unet_int8.onnx",
    quantize_mode="int8",
    high_precision_dtype="float32",
)