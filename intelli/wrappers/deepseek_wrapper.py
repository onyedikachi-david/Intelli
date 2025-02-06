import os
import json
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from intelli.model.deepseek.model import Transformer, ModelArgs
from intelli.model.deepseek.loader import ModelLoader
from intelli.model.deepseek.kernel import weight_dequant


class DeepSeekVariant(str, Enum):
    """Available DeepSeek model variants."""
    R1_671B = "deepseek-ai/deepseek-r1"
    R1_DISTILL_QWEN_1_5B = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
    R1_DISTILL_QWEN_7B = "deepseek-ai/deepseek-r1-distill-qwen-7b"
    R1_DISTILL_LLAMA_8B = "deepseek-ai/deepseek-r1-distill-llama-8b"
    R1_DISTILL_QWEN_14B = "deepseek-ai/deepseek-r1-distill-qwen-14b"
    R1_DISTILL_QWEN_32B = "deepseek-ai/deepseek-r1-distill-qwen-32b"
    R1_DISTILL_LLAMA_70B = "deepseek-ai/deepseek-r1-distill-llama-70b"


class DeepSeekWrapper:
    """High-level wrapper for DeepSeek model inference."""
    
    def __init__(
        self,
        model_variant: Union[str, DeepSeekVariant] = None,
        device: Optional[str] = None,
        quantize: bool = True,
        dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = True,
        max_memory: Optional[Dict[Union[int, str], str]] = None
    ):
        """
        Initialize DeepSeek wrapper.
        
        Args:
            model_variant: DeepSeek model variant to use. Can be model path, HF repo ID, or DeepSeekVariant.
                         Defaults to environment variable or smallest distilled model.
            device: Device to run model on. Defaults to CUDA if available.
            quantize: Whether to use quantization. Defaults to True for CUDA.
            dtype: Data type for model weights. Defaults to float16 for CUDA, float32 for CPU.
            use_flash_attention: Whether to use flash attention optimization. Defaults to True.
            max_memory: Maximum memory allocation per device. Format: {device: max_memory}
                       Example: {0: "10GiB", "cpu": "30GiB"}
        """
        # Set model variant
        if model_variant is None:
            model_variant = os.getenv('DEEPSEEK_MODEL_PATH', DeepSeekVariant.R1_DISTILL_QWEN_1_5B.value)
        elif isinstance(model_variant, DeepSeekVariant):
            model_variant = model_variant.value
            
        self.model_variant = model_variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantize = quantize and self.device == "cuda"
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.use_flash_attention = use_flash_attention
        self.max_memory = max_memory
        
        # Default generation parameters
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.max_length = 2048
        self.repetition_penalty = 1.1
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load model and tokenizer with optimizations."""
        loader = ModelLoader()
        
        # Determine if we need device map for large models
        device_map = None
        if self.max_memory is not None:
            device_map = "auto"
            
        # Load model with optimizations
        model_data = loader.load_model(
            self.model_variant,
            device=self.device,
            quantize=self.quantize,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            device_map=device_map,
            max_memory=self.max_memory
        )
        
        # Initialize model
        self.config = model_data["config"]
        self.tokenizer = model_data["tokenizer"]
        
        # Update config for model size
        if "model_type" in self.config:
            if "70b" in self.model_variant.lower():
                self.config["dim"] = 8192
                self.config["n_layers"] = 80
            elif "32b" in self.model_variant.lower():
                self.config["dim"] = 6144
                self.config["n_layers"] = 60
            elif "14b" in self.model_variant.lower():
                self.config["dim"] = 5120
                self.config["n_layers"] = 40
            elif "8b" in self.model_variant.lower() or "7b" in self.model_variant.lower():
                self.config["dim"] = 4096
                self.config["n_layers"] = 32
            elif "1.5b" in self.model_variant.lower():
                self.config["dim"] = 2048
                self.config["n_layers"] = 24
        
        args = ModelArgs(**self.config)
        self.model = Transformer(args).to(self.device)
        
        # Load weights with potential sharding for large models
        if device_map is not None:
            self.model.load_state_dict(model_data["weights"], strict=False)
        else:
            self.model.load_state_dict(model_data["weights"])
            
        self.model.eval()
        
    def update_params(self, **kwargs):
        """
        Update generation parameters.
        
        Args:
            **kwargs: Parameters to update (temperature, top_p, top_k, max_length, repetition_penalty)
        """
        valid_params = {
            'temperature', 'top_p', 'top_k', 'max_length', 'repetition_penalty'
        }
        
        for k, v in kwargs.items():
            if k in valid_params:
                setattr(self, k, v)
                
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Optional generation parameters to override defaults
            
        Returns:
            Generated text response
        """
        # Update parameters if provided
        if kwargs:
            temp_params = {k: getattr(self, k) for k in ['temperature', 'top_p', 'top_k', 'max_length', 'repetition_penalty']}
            self.update_params(**kwargs)
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Track generated tokens for repetition penalty
        generated = []
        
        # Generate tokens
        for _ in range(self.max_length):
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs[0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / self.temperature
                
                # Apply repetition penalty
                if len(generated) > 0:
                    for token in generated:
                        next_token_logits[token] /= self.repetition_penalty
                
                # Apply top-k filtering
                if self.top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, self.top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if self.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated tokens
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if end of text token is generated
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Restore original parameters if they were temporarily overridden
        if kwargs:
            self.update_params(**temp_params)
        
        # Decode and return generated text
        return self.tokenizer.decode(input_ids[0].tolist())
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Alias for generate method."""
        return self.generate(prompt, **kwargs)

    def update_model_params(self, model_params: Dict[str, Any]):
        """Update model parameters."""
        self.model_params = model_params 