#!/bin/bash

# Set CUDA environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA for better error tracking
export TORCH_CUDA_ARCH_LIST="8.0"  # Optimize for A100

# Run GPU test
echo "Running GPU inference test..."
python -u intelli/test/integration/test_gpu.py 