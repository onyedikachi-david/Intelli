#!/bin/bash

# Print system info
echo "System Information:"
uname -a
echo

# Check NVIDIA driver
echo "NVIDIA Driver Information:"
nvidia-smi || echo "nvidia-smi failed - please check NVIDIA drivers"
echo

# Set CUDA environment variables
echo "Setting up CUDA environment..."

# Try to find CUDA installation
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-12.2"
    "/usr/local/cuda-12.1"
    "/usr/local/cuda-12.0"
    "/usr/local/cuda-11.8"
)

for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        export CUDA_HOME="$path"
        export CUDA_ROOT="$path"  # Some applications use this instead
        export PATH="$CUDA_HOME/bin:$PATH"
        # Add both lib64 and lib directories to library path
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "Found CUDA at: $CUDA_HOME"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    echo "Warning: Could not find CUDA installation"
fi

# Add additional CUDA-related paths
if [ -d "/usr/lib/x86_64-linux-gnu" ]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
fi

# Add CUDA version-specific paths
if [ -d "/usr/local/cuda/targets/x86_64-linux" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
fi

# Set additional CUDA environment variables
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA for better error tracking
export TORCH_CUDA_ARCH_LIST="8.0"  # Optimize for A100
export TORCH_USE_CUDA_DSA=1  # Enable CUDA Dynamic Shared Memory
export NCCL_DEBUG=INFO  # Enable NCCL debugging
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Memory allocation settings

# Print current environment
echo -e "\nCUDA Environment:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_ROOT: $CUDA_ROOT"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify CUDA libraries
echo -e "\nChecking CUDA libraries..."
REQUIRED_LIBS=(
    "libcudart.so"
    "libcublas.so"
    "libcufft.so"
    "libcurand.so"
    "libcusolver.so"
    "libcusparse.so"
    "libnvToolsExt.so"
)

for lib in "${REQUIRED_LIBS[@]}"; do
    if [ -f "$CUDA_HOME/lib64/$lib" ]; then
        echo "Found $lib"
    else
        echo "Warning: $lib not found"
    fi
done

# Print PyTorch version and CUDA info
echo -e "\nPyTorch and CUDA Information:"
python -c "
import torch
import os
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')
print(f'Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
print(f'\nEnvironment variables:')
print(f'CUDA_HOME: {os.environ.get("CUDA_HOME", "Not set")}')
print(f'LD_LIBRARY_PATH: {os.environ.get("LD_LIBRARY_PATH", "Not set")}')
print(f'\nCUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')
print(f'Current CUDA device: {torch.cuda.current_device() if torch.cuda.is_available() else "N/A"}')
print(f'\nTorch CUDA build info:')
print(f'CUDA arch list: {torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else "N/A"}')
print(f'CUDA device capability: {torch.cuda.get_device_capability() if torch.cuda.is_available() else "N/A"}')
" || echo "Failed to get PyTorch info"

# Run GPU test
echo -e "\nRunning GPU inference test..."
python -u intelli/test/integration/test_gpu.py 