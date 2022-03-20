import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
from time import time

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

__global__ void conway_ker(int *pLattice, int iters)
{
    int x = _X, y = _Y;
    __shared__ int lattice[32*32];

    lattice[_INDEX(x,y)] = pLattice[_INDEX(x,y)];
    __syncthreads();

    for (int i = 0; i < iters; i++)
    {
        int n = nbrs(x, y, lattice);
        int cellValue;
        if (lattice[ _INDEX(x,y) ] == 1)
        {
            switch(n)
            {
                case 2:
                case 3: cellValue = 1;
                    break;
                default: cellValue = 0;
            }
        }
        else if (lattice[ _INDEX(x,y) ] == 0)
        {
            switch(n)
            {
                case 3: cellValue = 1;
                    break;
                default: cellValue = 0;
            }
        }
        __syncthreads();
        lattice[ _INDEX(x,y) ] = cellValue;
        __syncthreads();
    }

    __syncthreads();
    pLattice[_INDEX(x,y)] = lattice[_INDEX(x,y)];
    __syncthreads();
}
""")

myLife = ker.get_function("conway_ker")

if __name__ == '__main__':
    N = 32
    P = 0.3
    Itr = 10_000_000
    lattice = np.int32( np.random.choice([1,0], N*N, p=[P,1-P]).reshape(N,N) )

    t1 = time()
    lattice_gpu = gpuarray.to_gpu(lattice)
    t2 = time()
    myLife(lattice_gpu, np.int32(Itr), grid=(1,1,1), block=(32,32,1))
    t3 = time()
    latt_out = lattice_gpu.get()
    t4 = time()
    print(f'It took {(t3-t2)*1000:2.3f} ms to calculate {Itr} runs'
        + f' and {(t4-t3 + t2-t1):2.6f} seconds to transfer data.')

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(lattice)
    axes[1].imshow(latt_out)
    axes[0].set_title('Initial lattice')
    axes[1].set_title('Lattice after 1M runs')
    plt.show()