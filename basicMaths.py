import numpy as np
import pycuda.autoinit
from pycuda import gpuarray

host_data = np.array([1,2,3,4,5],dtype=np.float32)
dev_data = gpuarray.to_gpu(host_data)
dev_data_x2 = 2* dev_data
print(f'How it started: {host_data}')
host_data = dev_data_x2.get()
print(f'How it\'s going: {host_data}')
