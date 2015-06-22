#!/usr/bin/python
import arrayfire as af

af.info()
print(af.device_info())
print(af.get_device_count())
print(af.is_dbl_supported())
af.sync()

print('starting the loop')
for k in range(af.get_device_count()):
    af.set_device(k)
    dev = af.get_device()
    assert(k == dev)

    print(af.is_dbl_supported(k))

    a = af.randu(100, 100)
    af.sync(dev)
    mem_info = af.device_mem_info()
    assert(mem_info['alloc']['buffers'] == 1)
    assert(mem_info[ 'lock']['buffers'] == 1)
