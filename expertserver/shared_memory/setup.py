from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='shared_memory',
    ext_modules=[
        CUDAExtension('shared_memory', [
            'shared_memory.cpp',
            'shared_memory_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })