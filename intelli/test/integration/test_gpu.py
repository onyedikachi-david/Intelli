import sys
import os
import torch
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper, DeepSeekVariant

def check_nvidia_smi():
    """Check NVIDIA GPU status using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nNVIDIA-SMI Output:")
            print(result.stdout)
            return True
        return False
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
        return False

def check_cuda_setup():
    """Check CUDA setup and environment."""
    print("\nCUDA Setup Check:")
    
    # Check CUDA environment variables
    cuda_path = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA_HOME/CUDA_PATH: {cuda_path}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Check PyTorch CUDA setup
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Check NVCC if available
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version'], universal_newlines=True)
        print(f"\nNVCC version:\n{nvcc_output}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nNVCC not found in PATH")

def test_gpu_inference():
    """Test DeepSeek model inference on GPU with memory optimizations."""
    
    print("\nChecking GPU setup...")
    
    # Run diagnostic checks
    check_cuda_setup()
    nvidia_smi_available = check_nvidia_smi()
    
    if not torch.cuda.is_available():
        if nvidia_smi_available:
            print("\nWarning: NVIDIA GPU detected but CUDA is not available in PyTorch.")
            print("This might be due to:")
            print("1. PyTorch not compiled with CUDA support")
            print("2. Incompatible CUDA version")
            print("3. Missing CUDA runtime libraries")
            print("\nPlease ensure:")
            print("1. PyTorch is installed with CUDA support (e.g., pip install torch --index-url https://download.pytorch.org/whl/cu118)")
            print("2. CUDA toolkit is installed and in PATH")
            print("3. NVIDIA drivers are up to date")
        else:
            print("\nNo CUDA-capable GPU detected.")
            print("Please ensure NVIDIA drivers are installed and GPU is properly connected.")
        return
        
    # Get GPU info
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_memory_gb = total_memory / (1024**3)  # Convert to GB
    
    print(f"\nGPU Info:")
    print(f"Device: {gpu_name}")
    print(f"Total Memory: {total_memory_gb:.1f} GB")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Initialize with GPU device and memory constraints
    # Use about 75% of available GPU memory
    max_gpu_memory = f"{int(total_memory_gb * 0.75)}GiB"
    
    try:
        print("\nInitializing model...")
        model = DeepSeekWrapper(
            model_variant=DeepSeekVariant.R1_DISTILL_QWEN_1_5B,
            device="cuda",
            quantize=True,  # Enable quantization for memory efficiency
            dtype=torch.float16,  # Use half precision
            max_memory={"cuda": max_gpu_memory}
        )
    except Exception as e:
        print(f"\nError initializing model: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
        return
    
    # Test prompts of varying complexity
    prompts = [
        # Short prompt
        "What is the capital of France?",
        
        # Medium prompt
        """Explain the difference between supervised and unsupervised learning in machine learning.
        Include examples of each.""",
        
        # Long prompt with code
        """Write a Python function that implements merge sort with the following requirements:
        1. Should be type-hinted
        2. Include docstring with examples
        3. Handle edge cases
        4. Be memory efficient""",
        
        # Complex reasoning prompt
        """Consider a distributed system with multiple nodes processing transactions concurrently.
        What are the key challenges in maintaining consistency and how would you address them?
        Provide specific examples and solutions."""
    ]
    
    print("\nRunning inference tests...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest {i}/{len(prompts)}")
        print(f"Prompt length: {len(prompt)} chars")
        
        try:
            # Track GPU memory before generation
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated()
            
            # Generate response with slightly different parameters for each test
            response = model.generate(
                prompt,
                temperature=0.7 + (i * 0.1),  # Vary temperature
                top_p=0.9,
                max_length=min(100 * i, 2048),  # Vary length based on test
                repetition_penalty=1.1
            )
            
            # Get memory stats
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            print(f"\nMemory Usage:")
            print(f"Before: {memory_before / 1024**2:.1f} MB")
            print(f"After: {memory_after / 1024**2:.1f} MB")
            print(f"Peak: {peak_memory / 1024**2:.1f} MB")
            print(f"Response length: {len(response)} chars")
            
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            print("\nStack trace:")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Clear cache between tests
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test_gpu_inference() 