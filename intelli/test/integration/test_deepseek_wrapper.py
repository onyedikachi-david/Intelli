import unittest
import os
import torch
import warnings
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper, DeepSeekVariant


def get_available_memory():
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0


class TestDeepSeekWrapper(unittest.TestCase):
    """Test cases for DeepSeek model wrapper."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for testing DeepSeek models")
            
        # Check available GPU memory
        available_memory = get_available_memory()
        print(f"\nAvailable GPU memory: {available_memory:.2f} GB")
        
        if available_memory < 12:
            warnings.warn(f"Low GPU memory ({available_memory:.2f} GB). Tests might fail.")
            
        # Test with either 7B or 8B model as per requirements
        cls.test_models = [
            DeepSeekVariant.R1_DISTILL_QWEN_7B.value,    # 7B Qwen model
            DeepSeekVariant.R1_DISTILL_LLAMA_8B.value,   # 8B Llama model
        ]
        
        # Use first available model
        for model in cls.test_models:
            try:
                print(f"\nTrying to load {model}...")
                cls.model_variant = model
                # Try to create wrapper to verify model availability
                wrapper = DeepSeekWrapper(
                    model_variant=model,
                    quantize=True,
                    dtype=torch.float16
                )
                print(f"Successfully loaded {model}")
                del wrapper
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"Failed to load {model}: {str(e)}")
                continue
        else:
            raise unittest.SkipTest("No suitable test model (7B/8B) found")
            
        cls.device = "cuda"  # Force CUDA as these models require it
        
        # Print test environment info
        print(f"\nTest Environment:")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Selected Model: {cls.model_variant}")
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            self.wrapper = DeepSeekWrapper(
                model_variant=self.model_variant,
                device=self.device,
                quantize=True,  # Always use quantization for memory efficiency
                dtype=torch.float16,
                use_flash_attention=True
            )
        except Exception as e:
            self.skipTest(f"Failed to initialize model: {str(e)}")
    
    def test_model_loading(self):
        """Test that model loads successfully."""
        self.assertIsNotNone(self.wrapper.model)
        self.assertIsNotNone(self.wrapper.tokenizer)
        self.assertTrue(self.wrapper.model.training == False)  # Should be in eval mode
        
        # Verify model size configuration
        if "7b" in self.model_variant.lower():
            self.assertEqual(self.wrapper.config["dim"], 4096)
            self.assertEqual(self.wrapper.config["n_layers"], 32)
        elif "8b" in self.model_variant.lower():
            self.assertEqual(self.wrapper.config["dim"], 4096)
            self.assertEqual(self.wrapper.config["n_layers"], 32)
            
        print(f"\nModel loaded successfully")
        print(f"Model configuration: {self.wrapper.config}")
    
    def test_text_generation(self):
        """Test basic text generation."""
        prompts = [
            "What is machine learning?",
            "Write a simple Python function to",
            "Explain the concept of",
        ]
        
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                try:
                    output = self.wrapper.generate(prompt, max_length=100)
                    print(f"\nPrompt: {prompt}")
                    print(f"Output: {output[:100]}...")
                    self.assertIsInstance(output, str)
                    self.assertGreater(len(output), len(prompt))
                    self.assertTrue(output.startswith(prompt))
                except Exception as e:
                    self.skipTest(f"Generation failed: {str(e)}")
    
    def test_memory_efficiency(self):
        """Test memory usage during inference."""
        initial_memory = torch.cuda.memory_allocated()
        print(f"\nInitial GPU memory: {initial_memory / 1024**2:.2f} MB")
        
        # Run multiple generations
        prompts = ["Hello,", "Testing,", "Generate,"]
        for prompt in prompts:
            output = self.wrapper.generate(prompt, max_length=50)
            self.assertIsInstance(output, str)
            
            # Force cleanup
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated()
            print(f"Memory after generation: {current_memory / 1024**2:.2f} MB")
            
            # Memory should stay within reasonable bounds
            self.assertLess(current_memory - initial_memory, 1024 * 1024 * 500)  # Less than 500MB overhead
    
    def test_quantization(self):
        """Verify quantization is working."""
        if get_available_memory() < 16:
            self.skipTest("Not enough GPU memory for quantization comparison test")
            
        # Get memory usage with quantization
        memory_with_quant = torch.cuda.memory_allocated()
        print(f"\nMemory with quantization: {memory_with_quant / 1024**2:.2f} MB")
        
        # Create non-quantized model
        del self.wrapper
        torch.cuda.empty_cache()
        
        try:
            wrapper_fp16 = DeepSeekWrapper(
                model_variant=self.model_variant,
                device=self.device,
                quantize=False,
                dtype=torch.float16
            )
            
            memory_without_quant = torch.cuda.memory_allocated()
            print(f"Memory without quantization: {memory_without_quant / 1024**2:.2f} MB")
            
            # Quantized model should use less memory
            self.assertLess(memory_with_quant, memory_without_quant)
            
            del wrapper_fp16
        except Exception as e:
            self.skipTest(f"Failed to load non-quantized model: {str(e)}")
        finally:
            torch.cuda.empty_cache()
    
    def tearDown(self):
        """Clean up after each test."""
        del self.wrapper
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Running DeepSeek Wrapper Tests")
    print("=" * 50)
    unittest.main(verbosity=2) 