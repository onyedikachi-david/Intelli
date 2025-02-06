import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper, DeepSeekVariant

def test_cpu_inference():
    # Initialize with CPU device
    model = DeepSeekWrapper(
        model_variant=DeepSeekVariant.R1_DISTILL_QWEN_1_5B,  # Using smallest model for quick test
        device="cpu",
        quantize=False  # Disable quantization since we're using CPU
    )
    
    # Test prompt
    prompt = "Write a simple Python function to calculate fibonacci numbers."
    
    print("Model initialized, generating response...")
    response = model.generate(prompt)
    print("\nPrompt:", prompt)
    print("\nResponse:", response)

if __name__ == "__main__":
    test_cpu_inference() 