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

# Check MIG configuration
echo -e "\nChecking MIG Configuration:"
nvidia-smi mig -lgi || echo "Failed to list GPU instances"
nvidia-smi mig -lci || echo "Failed to list compute instances"

# Set MIG-specific environment variables
export CUDA_VISIBLE_DEVICES="MIG-GPU-0/0/0"  # Use first compute instance of first GPU instance
export CUDA_MIG_DEVICE_SCOPE="single"
export NVIDIA_MIG_CONFIG_DEVICES="all"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility,video"

# Set additional CUDA environment variables
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA for better error tracking
export TORCH_CUDA_ARCH_LIST="8.0"  # Optimize for A100
export TORCH_USE_CUDA_DSA=1  # Enable CUDA Dynamic Shared Memory
export NCCL_DEBUG=INFO  # Enable NCCL debugging
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Memory allocation settings

# Print current environment
echo -e "\nCUDA Environment:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_ROOT: $CUDA_ROOT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
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
python3 -c '
import torch
import os
import sys

def print_cuda_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}")
    
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        print(f"\nCUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device capability: {torch.cuda.get_device_capability()}")
        print(f"\nCUDA device properties:")
        props = torch.cuda.get_device_properties(0)
        print(f"  Name: {props.name}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multi processor count: {props.multi_processor_count}")
        print(f"  Max threads per block: {props.max_threads_per_block}")
        print(f"  Max threads per MP: {props.max_threads_per_multi_processor}")
    else:
        print("Using device: CPU")
        
    print(f"\nEnvironment variables:")
    print(f"CUDA_HOME: {os.environ.get("CUDA_HOME", "Not set")}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")}")
    print(f"LD_LIBRARY_PATH: {os.environ.get("LD_LIBRARY_PATH", "Not set")}")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory Info:")
        print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")

try:
    print_cuda_info()
except Exception as e:
    print(f"Error getting CUDA info: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
' || echo "Failed to get PyTorch info"

# Run GPU test
echo -e "\nRunning GPU inference test..."
python -u intelli/test/integration/test_gpu.py 