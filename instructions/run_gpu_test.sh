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
    "/usr/local/cuda-11.8"
    "/usr/local/cuda-12.0"
    "/usr/local/cuda-12.1"
)

for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        export CUDA_HOME="$path"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        echo "Found CUDA at: $CUDA_HOME"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    echo "Warning: Could not find CUDA installation"
fi

# Set CUDA device and optimization flags
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA for better error tracking
export TORCH_CUDA_ARCH_LIST="8.0"  # Optimize for A100

# Print PyTorch version and CUDA info
echo -e "\nPyTorch and CUDA Information:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')" || echo "Failed to get PyTorch info"

# Run GPU test
echo -e "\nRunning GPU inference test..."
python -u intelli/test/integration/test_gpu.py 