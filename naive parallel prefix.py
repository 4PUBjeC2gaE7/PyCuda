import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
from time import time

naive_ker = SourceModule("""
__global__ void naive_prefix(double *vec, double *out)
{
    __shared__ double sum_buf[1024];
    int tid = threadIdx.x;
    sum_buf[tid] = vec[tid];

    int iter = 1;
    for (int i=0; i<10; i++)
    {
        __syncthreads();
        if (tid >= iter)
        {
            sum_buf[tid] = sum_buf[tid] + sum_buf[tid - iter];
        }
        iter = iter << 1;
    }
    __syncthreads();
    out[tid] = sum_buf[tid];
    __syncthreads();
}
""")

naive_gpu = naive_ker.get_function("naive_prefix")

if __name__ == '__main__':
    testVec = np.random.randn(1024).astype(np.float64)
    testVec_gpu = gpuarray.to_gpu(testVec)

    outVec_gpu = gpuarray.empty_like(testVec_gpu)
    t1 = time()
    naive_gpu(testVec_gpu, outVec_gpu, block=(1024,1,1), grid=(1,1,1))
    t2 = time()
    totalSum = sum(testVec)
    t3 = time()
    totalSum_gpu = outVec_gpu[-1].get()

    isMatch = 'true' if np.allclose(totalSum, totalSum_gpu) else 'false'
    print(f'Data Match: {isMatch}')

    print(f'GPU time: {(t2 - t1)*1000:2.4f}ms')
    print(f'CPU time: {(t3 - t2)*1000:2.4f}ms')
