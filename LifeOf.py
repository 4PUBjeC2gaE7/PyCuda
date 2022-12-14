from configparser import Interpolation
from re import L
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

ker = SourceModule("""
#define _X ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y ( threadIdx.y + blockIdx.y * blockDim.y )
#define _WIDTH ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y )
#define _XM(x) ( (x + _WIDTH) % _WIDTH )
#define _YM(y) ( (y + _HEIGHT) % _HEIGHT )
#define _INDEX(x,y) ( _XM(x) + _YM(y) * _WIDTH )

__device__ int nbrs(int x, int y, int *in)
{
    return ( \
        in[ _INDEX(x-1,y-1) ] + in[ _INDEX(x-1,y) ] + in[ _INDEX(x-1,y+1) ] + \
        in[ _INDEX(x,  y-1) ] + in[ _INDEX(x,  y) ] + in[ _INDEX(x,  y+1) ] + \
        in[ _INDEX(x+1,y-1) ] + in[ _INDEX(x+1,y) ] + in[ _INDEX(x+1,y+1) ] \
    );
}

__global__ void conway_ker(int *latticeOut, int *latticeIn)
{
    int x = _X, y = _Y;
    int n = nbrs(x, y, latticeIn);

    if (latticeIn[ _INDEX(x,y) ] == 1)
    {
        switch(n)
        {
            case 2:
            case 3: latticeOut[ _INDEX(x,y) ] = 1;
                break;
            default: latticeOut[ _INDEX(x,y) ] = 0;
        }
    }
    else if (latticeIn[ _INDEX(x,y) ] == 0)
    {
        switch(n)
        {
            case 3: latticeOut[ _INDEX(x,y) ] = 1;
                break;
            default: latticeOut[ _INDEX(x,y) ] = 0;
        }
    }
}
""")

myLife = ker.get_function("conway_ker")

def updateGpu(frameNum, img, newLattice_gpu, lattice_gpu, N):
    myLife(newLattice_gpu, lattice_gpu, grid=(int(N/32), int(N/32), 1), block=(32,32,1))
    img.set_data(newLattice_gpu.get())
    lattice_gpu[:] = newLattice_gpu[:]
    return img

if __name__ == '__main__':
    N = 256
    P = 0.8
    lattice = np.int32( np.random.choice([0,1], N*N, p=[P,1-P]).reshape(N,N) )
    lattice_gpu = gpuarray.to_gpu(lattice)
    newLattice_gpu = gpuarray.empty_like(lattice_gpu)

    fig, ax = plt.subplots()
    img = ax.imshow(lattice_gpu.get(), interpolation='nearest')
    ani = anim.FuncAnimation(fig, updateGpu, fargs=(img, newLattice_gpu, lattice_gpu, N), interval=0, frames=1000, save_count=1000)
    plt.show()