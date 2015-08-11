from setuptools import setup
from distutils.extension import Extension
from numpy import get_include

setup(
    name='flattnr',
    version='0.1.dev0',
    packages=['flattnr'],
    entry_points={
        'console_scripts': [
            'flattnr_sim_cyl = flattnr.tools.flattnr_simulate_cylinder:main',
            'flattnr_edgels_plot = flattnr.tools.flattnr_edgels_plot:main',
            'flattnr_test_error = flattnr.tools.flattnr_test_error:main',
            'flattnr_extract_edgels = flattnr.tools.flattnr_extract_edgels:main'
        ]
    },

    install_requires=[
        # 'filtersqp',
        # 'cython',
        # 'PIL',
        'numpy'
    ],

    ext_modules=[
        Extension(
            'flattnr.lowlvl_target._core',
            sources=[
                'flattnr/lowlvl_target/_core_module.cpp',
            ],
            depends=[
                'flattnr/lowlvl_target/PyTupleStream.cpp',
                'flattnr/lowlvl_target/ArrayManager.cpp',
            ],
            # extra_compile_args=['-std=c++0x']
        ),
    ],
    include_dirs=[get_include()],
)
