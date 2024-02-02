from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='shared_pinned_memory',
    ext_modules=[
        CUDAExtension('shared_pinned_memory', [
            'shared_pinned_memory.cpp',
            'shared_pinned_memory_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })