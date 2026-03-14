import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.fspath(Path(self.get_ext_fullpath(ext.name)).parent.resolve())

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPT_BUILD_PYTHON=ON",
            "-DPT_USE_TUDA=ON",
            "-DPT_USE_CUDA=OFF",
            "-DPT_USE_LINQ=OFF",
            "-DPT_BUILD_TESTS=OFF",
            "-DPT_BUILD_SHARED_LIBS=ON",
        ]

        # Auto-detect CUDA
        if os.environ.get("PT_USE_CUDA", "0") == "1":
            cmake_args.append("-DPT_USE_CUDA=ON")

        # Auto-detect LinQ
        if os.environ.get("PT_USE_LINQ", "0") == "1":
            cmake_args.append("-DPT_USE_LINQ=ON")

        build_args = ["--config", "Release", f"-j{os.cpu_count()}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp, check=True
        )

setup(
    ext_modules=[CMakeExtension("promethorch._C")],
    cmdclass={"build_ext": CMakeBuild},
)
