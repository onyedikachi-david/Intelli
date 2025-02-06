import unittest
import os
import torch
from intelli.model.deepseek.loader import ModelLoader
from intelli.model.deepseek.model import Transformer, ModelArgs

class TestDeepSeekModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.model_path = os.getenv('DEEPSEEK_MODEL_PATH', 'deepseek-ai/deepseek-v3')
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.loader = ModelLoader(cache_dir="./test_cache")
        
    def test_model_loading(self):
        """Test basic model loading functionality."""
        model_data = self.loader.load_model(
            self.model_path,
            device=self.device,
            quantize=False
        )
        
        self.assertIsNotNone(model_data["config"])
        self.assertIsNotNone(model_data["tokenizer"])
        self.assertIsNotNone(model_data["weights"])
        
        # Test model instantiation
        args = ModelArgs(**model_data["config"])
        model = Transformer(args)
        model.load_state_dict(model_data["weights"])
        
    def test_quantized_loading(self):
        """Test loading quantized model version."""
        model_data = self.loader.load_model(
            self.model_path,
            device=self.device,
            quantize=True,
            dtype=torch.float16
        )
        
        # Check if weights are properly quantized
        for tensor in model_data["weights"].values():
            if hasattr(tensor, '_tensor'):
                tensor = tensor.materialize()
            self.assertTrue(tensor.dtype in [torch.float16, torch.int8])
            
    def test_memory_efficiency(self):
        """Test memory-efficient loading with lazy tensors."""
        model_data = self.loader.load_model(
            self.model_path,
            device=self.device
        )
        
        # Check lazy loading
        for name, tensor in model_data["weights"].items():
            self.assertFalse(hasattr(tensor, '_tensor'))
            # Access tensor to trigger materialization
            if hasattr(tensor, 'materialize'):
                _ = tensor.materialize()
                self.assertTrue(hasattr(tensor, '_tensor'))
            
    def test_model_inference(self):
        """Test model inference capabilities."""
        model_data = self.loader.load_model(
            self.model_path,
            device=self.device
        )
        
        args = ModelArgs(**model_data["config"])
        model = Transformer(args).to(self.device)
        model.load_state_dict(model_data["weights"])
        
        # Test tokenization and inference
        tokenizer = model_data["tokenizer"]
        test_input = "Hello, how are you?"
        tokens = tokenizer.encode(test_input)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tokens)
            
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[1], len(tokens[0]))  # Sequence length
        self.assertEqual(output.shape[2], args.vocab_size)  # Vocabulary size
        
    def test_model_generation(self):
        """Test text generation capabilities."""
        model_data = self.loader.load_model(
            self.model_path,
            device=self.device
        )
        
        args = ModelArgs(**model_data["config"])
        model = Transformer(args).to(self.device)
        model.load_state_dict(model_data["weights"])
        tokenizer = model_data["tokenizer"]
        
        def generate_text(prompt: str, max_length: int = 20, temperature: float = 0.7):
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = model(input_ids)
                    next_token_logits = outputs[0, -1, :] / temperature
                    next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                        
            return tokenizer.decode(input_ids[0].tolist())
        
        prompt = "Hello,"
        output = generate_text(prompt)
        self.assertIsInstance(output, str)
        self.assertTrue(output.startswith(prompt))
        self.assertGreater(len(output), len(prompt))
        
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists("./test_cache"):
            import shutil
            shutil.rmtree("./test_cache")
            
        torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main() 