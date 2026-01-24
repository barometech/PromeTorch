"""
PromeTorch Setup Script
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PYTHON_DIR = Path(__file__).parent.absolute()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DPT_BUILD_PYTHON=ON',
            '-DPT_BUILD_TESTS=OFF',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        # Check for CUDA
        cuda_available = os.environ.get('PT_USE_CUDA', '').lower() in ('1', 'on', 'true')
        if cuda_available:
            cmake_args += ['-DPT_USE_CUDA=ON']

            cuda_path = os.environ.get('CUDA_PATH', '')
            if cuda_path:
                cmake_args += [
                    f'-DCMAKE_CUDA_COMPILER={cuda_path}/bin/nvcc',
                    f'-DCUDAToolkit_ROOT={cuda_path}',
                ]

        if sys.platform.startswith('win'):
            cmake_args += ['-G', 'NMake Makefiles']

        build_args += ['--', '-j4']

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(['cmake', str(PROJECT_ROOT)] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)


# ============================================================================
# Setup
# ============================================================================

setup(
    name='promethorch',
    version='0.1.0',
    author='PromeTorch Team',
    author_email='',
    description='A PyTorch-like Deep Learning Framework',
    long_description=open(PROJECT_ROOT / 'README.md').read() if (PROJECT_ROOT / 'README.md').exists() else '',
    long_description_content_type='text/markdown',
    url='https://github.com/promethorch/promethorch',

    packages=find_packages(),
    package_dir={'': '.'},

    ext_modules=[CMakeExtension('promethorch._C', str(PROJECT_ROOT))],
    cmdclass={'build_ext': CMakeBuild},

    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
        ],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
