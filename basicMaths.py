import numpy as np
import pycuda.autoinit
from pycuda import gpuarray

host_data = np.array([1,2,3,4,5],dtype=np.float32)
dev_data = gpuarray.to_gpu(host_data)
dev_data_x2 = 2* dev_data
print(f'How it started: {host_data}')
host_data = dev_data_x2.get()
print(f'How it\'s going: {host_data}')

host_x = np.array([1,2,3,4], dtype=np.float32)
host_y = np.array([3,2,1,0], dtype=np.float32)
host_z = np.array([1,1,1,1], dtype=np.float32)

dev_x = gpuarray.to_gpu(host_x)
dev_y = gpuarray.to_gpu(host_y)
dev_z = gpuarray.to_gpu(host_z)

print(f'{"Function":<20}{"HOST":<20}{"DEVICE":<20}')
print(f'{"x + y":<20}{str(host_x + host_y):<20}{(dev_x + dev_y).get()}')
print(f'{"x ^ z":<20}{str(host_x ** host_z):<20}{(dev_x ** dev_z).get()}')
print(f'{"x / x":<20}{str(host_x / host_x):<20}{(dev_x / dev_x).get()}')
print(f'{"z - x":<20}{str(host_z - host_x):<20}{(dev_z - dev_x).get()}')
print(f'{"z^2":<20}{str(host_z**2):<20}{(dev_z**2).get()}')
print(f'{"x - 1":<20}{str(host_x - 1):<20}{(dev_x - 1).get()}')
