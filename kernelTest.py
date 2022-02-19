import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

ker = SourceModule("""
__global__ void scalarMultiplyKernel(float *vecIn, float scalar, float *vecOut)
{
    int i = threadIdx.x;
    vecOut[i] = vecIn[i] * scalar;
}
""")

myKernel = ker.get_function("scalarMultiplyKernel")
testVec = np.random.randn(512).astype(np.float32)
testVec_gpu = gpuarray.to_gpu(testVec)
outVec_gpu = gpuarray.empty_like(testVec_gpu)

myKernel(testVec_gpu, np.float32(2), outVec_gpu, block = (512,1,1), grid = (1,1,1))

isMatch = 'true' if np.allclose(outVec_gpu.get(), testVec * 2) else 'false'
print(f'Data Match: {isMatch}')
