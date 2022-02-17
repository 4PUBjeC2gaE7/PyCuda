import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def simpleMandelbrot(width, height, rLow, rHigh, iLow, iHigh, maxIter):
    reVals = np.linspace(rLow, rHigh, width)
    imVals = np.linspace(rLow, rHigh, width)
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

if __name__ == '__main__':
    t1 = time()
    mandel = simpleMandelbrot(512,512,-2,2,-2,2,256)
    t2 = time()
    print(f'It took {t2 - t1} seconds to calculate the Mandelbrot')

    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2,2,-2,2))
    plt.savefig('mandelbrot.png', dpi=fig.dpi)
    t2 = time()
    print(f'It took {t2 - t1} seconds to dump the image')