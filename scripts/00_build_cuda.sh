#!/bin/bash
# Compile CUDA stress kernels for MLWD sensitivity collection
set -e
cd "$(dirname "$0")/.."

mkdir -p build/cuda

echo "Compiling CUDA stress kernels..."

# Detect GPU architecture
GPU_ARCH=${GPU_ARCH:-sm_70}  # V100 = sm_70

# Compile kernels.cu -> object
nvcc -c cuda/kernels.cu -o build/cuda/kernels.o \
    -arch=${GPU_ARCH} -Xcompiler -fPIC -Icuda/

# Compile stress_interface.cu -> shared library
nvcc -shared cuda/stress_interface.cu build/cuda/kernels.o \
    -o build/cuda/libstress_interface.so \
    -arch=${GPU_ARCH} -Xcompiler -fPIC -Icuda/

echo "Done: build/cuda/libstress_interface.so"
ls -lh build/cuda/libstress_interface.so
