from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_enqueue',
    ext_modules=[
        CUDAExtension('cuda_enqueue', [
            'enqueue.cpp',
            'cuda_enqueue.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })