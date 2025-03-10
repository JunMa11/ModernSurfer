import sys
sys.path.insert(0, './TensorRT-Model-Optimizer')
from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import OnnxBytes
# Configure deployment
deployment = {
    "runtime": "TRT",
    "version": "10.3",
    "precision": 'int8',
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

