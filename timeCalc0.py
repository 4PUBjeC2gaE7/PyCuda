import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time

print(f'{"Run":<6}{"Host [ms]":<12}{"Device [ms]":<12}{"Match?":>6}')

for i in range(25):
    host_data = np.float32(np.random.random(100_000_000))

    ht1 = time()
    host_data_2x = host_data * np.float32(2)
    ht2 = time()
    
    device_data = gpuarray.to_gpu(host_data)
    dt1 = time()
    device_data_2x = device_data * np.float32(2)
    dt2 = time()
    from_device = device_data_2x.get()

    isMatch = 'true' if np.allclose(from_device, host_data_2x) else 'false'

    print(f'{i:<6d}{(ht2 - ht1)*1000:>8.2f}   {(dt2 - dt1)*1000:>8.2f}    {isMatch:>6}')