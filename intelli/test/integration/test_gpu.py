import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper, DeepSeekVariant

def test_gpu_inference():
    """Test DeepSeek model inference on GPU with memory optimizations."""
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return
        
    # Get GPU info
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_memory_gb = total_memory / (1024**3)  # Convert to GB
    
    print(f"\nGPU Info:")
    print(f"Device: {gpu_name}")
    print(f"Total Memory: {total_memory_gb:.1f} GB")
    
    # Initialize with GPU device and memory constraints
    # Use about 75% of available GPU memory
    max_gpu_memory = f"{int(total_memory_gb * 0.75)}GiB"
    
    model = DeepSeekWrapper(
        model_variant=DeepSeekVariant.R1_DISTILL_QWEN_1_5B,
        device="cuda",
        quantize=True,  # Enable quantization for memory efficiency
        dtype=torch.float16,  # Use half precision
        max_memory={"cuda": max_gpu_memory}
    )
    
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
        
        # Clear cache between tests
        torch.cuda.empty_cache()

if __name__ == "__main__":
    test_gpu_inference() 