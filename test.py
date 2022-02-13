import imp


import pycuda.driver as drv

drv.init()

print (f'Detected {drv.Device.count()} CUDA capable devices\n')

for i in range (drv.Device.count()):
    gpu_device = drv.Device(i)
    print(f'Device {i}: {gpu_device.name()}')
    compute_capability = float( '%d.%d' % gpu_device.compute_capability())
    print(f'\tComputer Capability: {compute_capability}')
    print(f'\tTotal Memory: {gpu_device.total_memory() // 1024**2} MB')
    device_attrs = gpu_device.get_attributes()
    for key, attr in device_attrs.items():
        print(f'{str(key):>40}: {attr}')
    num_mp = gpu_device.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)