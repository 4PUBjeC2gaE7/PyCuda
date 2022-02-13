import imp


import pycuda.driver as drv

drv.init()

print ('Detected {} CUDA capable devices'.format(drv.Device.count()))

for i in range (drv.Device.count()):
    gpu_device = drv.Device(i)
    print(f'Device {i}: {gpu_device.name()}')
    compute_capability = float( '%d.%d' % gpu_device.compute_capability())
    print(f'\tComputer Capability: {compute_capability}')
    print(f'\tTotal Memory: {gpu_device.total_memory() // 1024**2} MB')
