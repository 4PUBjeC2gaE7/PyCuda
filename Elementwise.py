import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel as EWK

host_data = np.float32(np.random.random(250_000_000))

gpu_2x_ker = EWK(
    "float *in, float *out",
    "out[i] = (2 * in[i]) + 3;",
    "gpu_2x_ker"
)

def speedCompare():
    t1 = time()
    host_data_2x = (host_data * np.float32(2)) + np.float32(3)
    t2 = time()
    print(f'CPU Time:{(t2 - t1)*1000:0.6f}ms')
    
    device_data = gpuarray.to_gpu(host_data)
    device_data_2x = gpuarray.empty_like(device_data)

    t1 = time()
    gpu_2x_ker(device_data, device_data_2x)
    t2 = time()
    print(f'GPU Time:{(t2 - t1)*1000:0.6f}ms')

    from_device = device_data_2x.get()
    isMatch = 'true' if np.allclose(from_device, host_data_2x) else 'false'
    print(f'Data Match: {isMatch}')

if __name__ == '__main__':
    for i in range(10):
        print(f'Run {i:02d}')
        speedCompare()