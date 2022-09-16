# pyCuda
Experimenting with [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone) processing.
These projects follow "[Hands on GPU Programming with Python and CUDA](https://www.packtpub.com/product/hands-on-gpu-programming-with-python-and-cuda/9781788993913)" by Dr. Brian Tuomanen.

## Contents
 Filepath                 |  File Size (Bytes) | Description 
 ------------------------ | ------------------:| -----------------------------------
basicMaths.py             |  1030 | Simple project timing the performance of math operations
deviceQuery.py            |   667 | Instantiates the CUDA engine an prints GPU info
Elementwise.py            |   985 | Simple project timing the performance of compound math operations
kernelTest.py             |   717 | Instatiates and runs a Kernel
LICENSE                   | 35823 | GNU GPLv3 License
LifeOf.py                 |  2331 | Conway's Game of Life
LifeOf_streams.py         |  3153 | Conway's Game of Life using streams
LifeOf_syncThreads.py     |  2723 | Conway's Game of Life using threads
mandelbrot.py             |  2973 | Generates a Mandelbrot fractal
multi-kernel stream.py    |  1576 | Runs multiple Kernels using streams
multi-kernel.py           |  1441 | Runs multiple Kernels
naive parallel prefix.py  |  3239 | Test of a basic parallel prefix scan
README.md                 |  1460 | This file
timeCalc0.py              |   682 | Basic test comparing GPU and CPU performance