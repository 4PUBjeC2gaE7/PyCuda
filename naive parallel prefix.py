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

up_ker = SourceModule("""
__global__ void up_ker(double *x, double *x_old, int k)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int _2k = 1 << k;
    int _2k1 = 1 << (k+1);
    int j = tid * _2k1;

    x[j + _2k1 - 1] = x_old[j + _2k - 1] + x_old[j + _2k1 - 1];
}
""")

down_ker = SourceModule("""
__global__ void down_ker(double *y, double *y_old, int k)
{
    int j = blockIdx.x*blockDim.x + threadIdx.x;

    int _2k = 1 << k;
    int _2k1 = 1 << (k+1);
    int j = tid * _2k1;

    y[j + _2k - 1] = y_old[j + _2k1 - 1];
    y[j + _2k1 - 1] = y_old[j + _2k1 - 1] + y_old[j + _2k - 1];
}
""")

naive_gpu = naive_ker.get_function("naive_prefix")
up_gpu = up_ker.get_function("up_ker")
dn_gpu = down_ker.get_function("down_ker")

def up_sweep(x):
    x = np.float64(x)
    x_gpu = gpuarray.to_gpu(np.float64(x))
    x_old_gpu = x_gpu.copy()
    for k in range ( int(np.log2(x.size))):
        num_threads = int(np.ceil(x.size / 2**(k+1)))
        grid_size = int(np.ceil(num_threads / 32))
        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads
        up_gpu(x_gpu, x_old_gpu, np.int32(k), block=(block_size,1,1), grid=(grid_size,1,1))
        x_old_gpu[:] = x_gpu[:]
    return(x_gpu.get())

def dn_sweep(y):
    y = np.float64(y)
    y[-1] = 0
    y_gpu = gpuarray.to_gpu(np.float64(y))
    y_old_gpu = y_gpu.copy()
    for k in range ( int(np.log2(y.size))):
        num_threads = int(np.ceil(y.size / 2**(k+1)))
        grid_size = int(np.ceil(num_threads / 32))
        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads
        dn_gpu(y_gpu, y_old_gpu, np.int32(k), block=(block_size,1,1), grid=(grid_size,1,1))
        y_old_gpu[:] = y_gpu[:]
    return(y_gpu.get())

def efficient_prefix(x):
    return(dn_sweep(up_sweep(x)))

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

    efficient_prefix(1024)