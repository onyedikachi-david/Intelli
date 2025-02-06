import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper, DeepSeekVariant

def test_cpu_inference():
    # Initialize with CPU device
    model = DeepSeekWrapper(
        model_variant=DeepSeekVariant.R1_DISTILL_QWEN_1_5B,
        device="cpu",
        quantize=False,
        max_memory={"cpu": "8GiB"}  # Add memory constraint
    )
    
    # Test with shorter prompt and length
    prompt = "Python fibonacci function:"
    print("Model initialized, generating response...")
    
    # Set conservative generation params
    response = model.generate(
        prompt,
        temperature=0.9,
        top_p=0.95,
        max_length=100  # Reduced from 2048 for testing
    )
    
    print("\nPrompt:", prompt)
    print("\nResponse:", response)

if __name__ == "__main__":
    test_cpu_inference() 