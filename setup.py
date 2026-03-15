"""
PromeTorch — Russian hardware-native deep learning framework.
Build with: pip install -e .
With CUDA:  PT_USE_CUDA=1 pip install -e .
With LinQ:  PT_USE_LINQ=1 pip install -e .
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # Find cmake
        cmake = shutil.which("cmake")
        if not cmake and platform.system() == "Windows":
            # Try common Windows cmake locations
            for candidate in [
                r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
                r"C:\Program Files\CMake\bin\cmake.exe",
                r"C:\ProgramData\anaconda3\Scripts\cmake.exe",
            ]:
                if Path(candidate).exists():
                    cmake = candidate
                    break
        if not cmake:
            print("WARNING: cmake not found — skipping C++ extension build.")
            print("Python package will work in fallback mode.")
            return

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        # _C goes into promethorch/
        extdir = extdir / "promethorch"
        extdir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPT_BUILD_PYTHON=ON",
            "-DPT_USE_TUDA=ON",
            "-DPT_USE_CUDA=OFF",
            "-DPT_USE_LINQ=OFF",
            "-DPT_USE_NMCARD=OFF",
            "-DPT_BUILD_TESTS=OFF",
        ]

        # Windows: use NMake or Ninja if available
        if platform.system() == "Windows":
            if shutil.which("ninja"):
                cmake_args.append("-GNinja")
            else:
                cmake_args.extend(["-G", "NMake Makefiles"])
            cmake_args.append("-DPT_BUILD_SHARED_LIBS=ON")
        else:
            cmake_args.append("-DPT_BUILD_SHARED_LIBS=ON")

        # Environment overrides
        if os.environ.get("PT_USE_CUDA", "0") == "1":
            cmake_args.append("-DPT_USE_CUDA=ON")
        if os.environ.get("PT_USE_LINQ", "0") == "1":
            cmake_args.append("-DPT_USE_LINQ=ON")
        if os.environ.get("PT_USE_NMCARD", "0") == "1":
            cmake_args.append("-DPT_USE_NMCARD=ON")

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # Configure
        result = subprocess.run(
            [cmake, ext.sourcedir, *cmake_args],
            cwd=build_temp, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"WARNING: cmake configure failed:\n{result.stderr[-500:]}")
            print("Skipping C++ build. Python package will work in fallback mode.")
            return

        # Build only _C target (not everything)
        build_args = [cmake, "--build", ".", "--target", "_C", "--config", "Release"]
        ncpu = os.cpu_count() or 1
        build_args.extend(["-j", str(ncpu)])

        result = subprocess.run(build_args, cwd=build_temp, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"WARNING: cmake build failed:\n{result.stderr[-500:]}")
            print("Skipping C++ build. Python package will work in fallback mode.")
            return

    def get_ext_filename(self, ext_name):
        """Override to prevent error when extension wasn't built."""
        return super().get_ext_filename(ext_name)

    def copy_extensions_to_source(self):
        """Override to skip copy if extension wasn't built."""
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            src = os.path.join(self.build_lib, filename)
            if not os.path.exists(src):
                print("Note: C++ extension not built, using Python-only mode.")
                return
        super().copy_extensions_to_source()

        # Also build c10 shared lib if needed
        c10_src = build_temp
        for pattern in ["c10.dll", "c10.so", "libc10.so", "libc10.dylib"]:
            for f in c10_src.rglob(pattern):
                dst = extdir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)


setup(
    name="promethorch",
    version="0.1.0",
    description="Russian hardware-native deep learning framework",
    author="PromeTorch Team",
    python_requires=">=3.8",
    packages=["promethorch", "promethorch.nn", "promethorch.optim"],
    install_requires=["numpy>=1.20"],
    ext_modules=[CMakeExtension("promethorch._C")],
    cmdclass={"build_ext": CMakeBuild},
)
