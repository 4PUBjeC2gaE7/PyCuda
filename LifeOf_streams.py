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

def updateGpu(frameNum, images, new_gpu, old_gpu, size, streams):
    for k in range(len(streams)):
        myLife( new_gpu[k],
                old_gpu[k],
                grid=(int(size/32), int(size/32), 1),
                block=(32,32,1),
                stream=streams[k],
                )
        images[k].set_data(new_gpu[k].get_async(stream=streams[k]))
        old_gpu[k].set_async(new_gpu[k], stream=streams[k])
    return images

if __name__ == '__main__':
    N = 256     # frame size, in cells
    P = 0.8     # probability of cell starting alive
    Sx = 8      # number of 'x' frames
    Sy = 3      # number of 'y' frames

    streams = []
    lattices_gpu = []
    newLattices_gpu = []
    imgs = []

    fig, ax = plt.subplots(nrows = Sy, ncols=Sx,figsize=(3*Sx,3*Sy))
    plt.tight_layout(h_pad=None, w_pad=None)

    for j in range(Sx):
        for k in range(Sy):
            streams.append(drv.Stream())
            lattice = np.int32( np.random.choice([0,1], N*N, p=[P,1-P]).reshape(N,N) )
            lattices_gpu.append(gpuarray.to_gpu(lattice))
            newLattices_gpu.append(gpuarray.empty_like(lattices_gpu[k]))
            imgs.append( ax[k,j].imshow( lattices_gpu[k].get_async(stream=streams[k]),
                                       interpolation='nearest'))
            ax[k,j].axis('off')
    
    ani = anim.FuncAnimation(fig, updateGpu, interval=0, frames=1000, save_count=1000,
        fargs=(imgs, newLattices_gpu, lattices_gpu, N, streams),
        )
    plt.show()