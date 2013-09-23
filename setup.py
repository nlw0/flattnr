from setuptools import setup
from distutils.extension import Extension

setup(
    name='flattnr',
    version='0.1dev',
    packages=['flattnr'],
    entry_points={
        'console_scripts': [
            'flattnr_sim_cyl = flattnr.tools.flattnr_simulate_cylinder:main',
            'flattnr_extract = flattnr.tools.flattnr_extract_edgels:main',
        ]
    },

    # tests_require=['Attest'],
    # test_loader='attest:Loader',
    # test_suite='tests.collection',

    install_requires=[
        # 'filtersqp',
        # 'cython',
        # 'PIL',
        'numpy'
    ],

    # cmdclass = {'build_ext': build_ext},
    # ext_modules = ext_modules,
)
