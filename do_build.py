"""Build the _C pybind11 module using subprocess to invoke nmake."""
import subprocess
import os
import sys

# Set up MSVC environment
msvc_bin = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
winsdk_bin = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64"
msvc_include = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\include"
winsdk_include = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0"
msvc_lib = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\lib\x64"
winsdk_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0"

env = os.environ.copy()
env["INCLUDE"] = f"{msvc_include};{winsdk_include}\\ucrt;{winsdk_include}\\shared;{winsdk_include}\\um;{winsdk_include}\\winrt"
env["LIB"] = f"{msvc_lib};{winsdk_lib}\\ucrt\\x64;{winsdk_lib}\\um\\x64"
env["PATH"] = f"{msvc_bin};{winsdk_bin};C:\\ProgramData\\anaconda3;C:\\ProgramData\\anaconda3\\Library\\bin;" + env.get("PATH", "")

build_dir = r"C:\Users\paper\Desktop\promethorch\build_pybind"

# Delete old obj files to force recompilation
obj_dir = os.path.join(build_dir, "CMakeFiles", "_C.dir", "python", "csrc")
if os.path.exists(obj_dir):
    for f in os.listdir(obj_dir):
        if f.endswith(".obj"):
            os.remove(os.path.join(obj_dir, f))
            print(f"Removed: {f}")

# Run nmake
nmake = os.path.join(msvc_bin, "nmake.exe")
print(f"Running: {nmake} _C")
print(f"In: {build_dir}")
result = subprocess.run(
    [nmake, "_C"],
    cwd=build_dir,
    env=env,
    capture_output=True,
    text=True,
    timeout=300
)

print("=== STDOUT ===")
print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
print("=== STDERR ===")
print(result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
print(f"=== EXIT CODE: {result.returncode} ===")
