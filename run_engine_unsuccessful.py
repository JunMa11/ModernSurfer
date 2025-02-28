import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

f = open("onnx_models/quant_fast_unet_fp16.engine", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

import numpy as np
BATCH_SIZE = 1
target_dtype = np.float16
# need to set input and output precisions to FP16 to fully enable it
output = np.empty([14, 64, 256, 256], dtype = target_dtype)

input_batch = np.array([1, 14, 64, 256, 256], dtype=target_dtype)

# allocate device memory
d_input = cuda.mem_alloc(2 * input_batch.nbytes)
d_output = cuda.mem_alloc(2 * output.nbytes)

tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
assert(len(tensor_names) == 2)

context.set_tensor_address(tensor_names[0], int(d_input))
context.set_tensor_address(tensor_names[1], int(d_output))

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()


def predict(batch):  # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v3(stream.handle)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    return output

pred = predict(input_batch)