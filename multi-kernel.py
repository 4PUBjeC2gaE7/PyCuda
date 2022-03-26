import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

num_arrays = 200
array_len = 1024**2

ker = SourceModule("""
__global__ void mult_ker(float *array, int array_len)
{
    int thd = blockIdx.x*blockDim.x + threadIdx.x;
    int iters = array_len / blockDim.x;

    for(int j=0; j<iters; j++)
    {
        int i = j*blockDim.x + thd;
        for(int k=0; k<50; k++)
        {
            array[i] *= 2.0;
            array[i] /= 2.0;
        }
    }
}
""")

mult_ker = ker.get_function('mult_ker')

if __name__ == '__main__':
    data = []
    data_gpu = []
    output_gpu = []

    for _ in range(num_arrays):
        data.append(np.random.randn(array_len).astype('float32'))

    t1 = time()
    for k in range(num_arrays):
        data_gpu.append(gpuarray.to_gpu(data[k]))
    t2 = time()
    for k in range(num_arrays):
        mult_ker(data_gpu[k], np.int32(array_len), block=(64,1,1), grid=(1,1,1))
    t3 = time()
    for k in range(num_arrays):
        output_gpu.append(data_gpu[k].get())
    t4 = time()

    for k in range(num_arrays):
        assert (np.allclose(output_gpu[k],data[k])),f"array item {k}"
    
    print(f'Copy to GPU: {(t2 - t1)*1000:>9.4f}ms')
    print(f'Calculation: {(t3 - t2)*1000:>9.4f}ms')
    print(f'Copy to MEM: {(t4 - t3)*1000:>9.4f}ms')