import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel as EWK
from time import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def simpleMandelbrot(width, height, reLow, reHigh, imLow, imHigh, maxIter):
    reVals = np.linspace(reLow, reHigh, width)
    imVals = np.linspace(reLow, reHigh, width)
    # we will represent members as 1, non-members as 0
    mandelbrotGraph = np.ones((height,width), dtype=np.float32)
    for x in range(width):
        for y in range(height):
            c = np.complex64(reVals[x] + imVals[y] * 1j)
            z = np.complex64(0)
            for i in range(maxIter):
                z = z**2 + c
                if(np.abs(z) > 2):
                    mandelbrotGraph[y,x] = 0
                    break
    return mandelbrotGraph

def gpuMandelbrot(width, height, reLow, reHigh, imLow, imHigh, maxIter, upperBound):
    reVals = np.matrix(np.linspace(reLow, reHigh,width), dtype=np.complex64)
    imVals = np.matrix(np.linspace(imLow, imHigh,height), dtype=np.complex64) * 1j
    mandelLattice = np.array(reVals + imVals.transpose(), dtype=np.complex64)
    # copy complex lattice to the GPU
    mandelLattice_gpu = gpuarray.to_gpu_async(mandelLattice)
    pycuda.autoinit.context.synchronize()
    # allocate output array on GPU
    mandelbrotGraph_gpu = gpuarray.empty(shape=mandelLattice.shape, dtype=np.float32)
    # run GPU kernel
    mandelKernel(mandelLattice_gpu, mandelbrotGraph_gpu, np.int32(maxIter), np.float32(upperBound))
    pycuda.autoinit.context.synchronize()
    # get data from GPU
    mandelbrotGraph = mandelbrotGraph_gpu.get_async()
    pycuda.autoinit.context.synchronize()
    return mandelbrotGraph

mandelKernel= EWK(
    "pycuda::complex<float> *lattice, float *mandelbrotGraph, int maxIters, float upperBound",
    """
    mandelbrotGraph[i] = 1;
    pycuda::complex<float> c = lattice[i];
    pycuda::complex<float> z(0,0);
    for (int j=0; j<maxIters; j++)
    {
        z = z*z + c;
        if(abs(z) > upperBound)
        {
            mandelbrotGraph[i] = 0;
            break;
        }
    }
    """,
    "mandelbrot_kernel"
)

if __name__ == '__main__':
    mySize = 2048
    myDepth = 2048
    myRange = 2

    # t1 = time()
    # mandel = simpleMandelbrot(mySize,mySize,-myRange,myRange,-myRange,myRange,myDepth)
    # t2 = time()
    # print(f'It took {t2 - t1} seconds to calculate on the Mandelbrot')

    t1 = time()
    mandel = gpuMandelbrot(mySize,mySize,-myRange,myRange,-myRange,myRange,myDepth,2)
    t2 = time()
    print(f'It took {t2 - t1} seconds to calculate on the Mandelbrot')

    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-myRange,myRange,-myRange,myRange))
    plt.savefig('mandelbrot.png', dpi=4000)
    t2 = time()
    print(f'It took {t2 - t1} seconds to dump the image')