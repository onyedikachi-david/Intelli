import os
import json
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from intelli.model.deepseek.model import Transformer, ModelArgs
from intelli.model.deepseek.loader import ModelLoader, LazyTensor
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
        
        # Download model files if needed
        model_path = loader.download_from_hf(self.model_variant)
        
        # Load model with optimizations
        model_data = loader.load_model(
            model_path,
            device=self.device,
            quantize=self.quantize,
            dtype=self.dtype
        )
        
        # Initialize model
        self.config = model_data["config"]
        self.tokenizer = model_data["tokenizer"]
        
        # Map config keys to expected names
        config_mapping = {
            'hidden_size': 'dim',
            'num_hidden_layers': 'n_layers',
            'num_attention_heads': 'n_heads',
            'intermediate_size': 'inter_dim',
            'max_sequence_length': 'max_seq_len'
        }
        
        # Filter config to only include expected arguments
        model_config = {}
        expected_args = [
            'dim', 'n_layers', 'n_heads', 'vocab_size', 'max_batch_size', 'max_seq_len',
            'dtype', 'inter_dim', 'moe_inter_dim', 'n_dense_layers', 'n_routed_experts',
            'n_shared_experts', 'n_activated_experts', 'n_expert_groups', 'n_limited_groups',
            'score_func', 'route_scale', 'q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim',
            'qk_rope_head_dim', 'v_head_dim', 'original_seq_len', 'rope_theta', 'rope_factor',
            'beta_fast', 'beta_slow', 'mscale'
        ]
        
        # First try direct mapping
        for key in expected_args:
            if key in self.config:
                model_config[key] = self.config[key]
            # Try mapped key names
            elif key in config_mapping.values():
                for old_key, new_key in config_mapping.items():
                    if new_key == key and old_key in self.config:
                        model_config[key] = self.config[old_key]
        
        # Update config for model size
        if "model_type" in self.config:
            if "70b" in self.model_variant.lower():
                model_config.update({
                    "dim": 8192,
                    "n_layers": 80,
                    "n_heads": 64,
                    "vocab_size": 151936,
                    "max_seq_len": 8192,
                    "max_batch_size": 32,
                    "inter_dim": 24576
                })
            elif "32b" in self.model_variant.lower():
                model_config.update({
                    "dim": 6144,
                    "n_layers": 60,
                    "n_heads": 48,
                    "vocab_size": 151936,
                    "max_seq_len": 8192,
                    "max_batch_size": 32,
                    "inter_dim": 18432
                })
            elif "14b" in self.model_variant.lower():
                model_config.update({
                    "dim": 5120,
                    "n_layers": 40,
                    "n_heads": 40,
                    "vocab_size": 151936,
                    "max_seq_len": 8192,
                    "max_batch_size": 32,
                    "inter_dim": 15360
                })
            elif "8b" in self.model_variant.lower() or "7b" in self.model_variant.lower():
                model_config.update({
                    "dim": 4096,
                    "n_layers": 32,
                    "n_heads": 32,
                    "vocab_size": 151936,
                    "max_seq_len": 8192,
                    "max_batch_size": 32,
                    "inter_dim": 12288
                })
            elif "1.5b" in self.model_variant.lower():
                model_config.update({
                    "dim": 1536,
                    "n_layers": 28,
                    "n_heads": 12,
                    "vocab_size": 151936,
                    "max_seq_len": 8192,
                    "max_batch_size": 32,
                    "inter_dim": 8960,
                    "kv_lora_rank": 256  # Critical: Set correct LoRA rank for 1.5B model
                })
        
        # Set default values for missing arguments
        defaults = {
            'dtype': 'bf16',
            'max_batch_size': 32,
            'max_seq_len': 8192,
            'inter_dim': 8960,  # Updated for 1.5B model
            'moe_inter_dim': 1408,
            'n_dense_layers': 1,
            'n_routed_experts': 64,
            'n_shared_experts': 2,
            'n_activated_experts': 6,
            'n_expert_groups': 1,
            'n_limited_groups': 1,
            'score_func': 'softmax',
            'route_scale': 1.0,
            'q_lora_rank': 0,
            'kv_lora_rank': 256,  # Critical: Default to 256 for LoRA rank
            'qk_nope_head_dim': 128,
            'qk_rope_head_dim': 64,
            'v_head_dim': 128,
            'original_seq_len': 4096,
            'rope_theta': 10000.0,
            'rope_factor': 40,
            'beta_fast': 32,
            'beta_slow': 1,
            'mscale': 1.0
        }
        for key, value in defaults.items():
            if key not in model_config:
                model_config[key] = value
        
        args = ModelArgs(**model_config)
        self.model = Transformer(args).to(self.device)
        
        # Convert state dict
        state_dict = {}
        for key, tensor in model_data["weights"].items():
            if isinstance(tensor, LazyTensor):
                tensor = tensor.materialize()
            # Remove "model." prefix from key
            if key.startswith("model."):
                key = key[6:]  # Remove "model." prefix
            
            # Handle key/value projection weights
            if 'k_proj.weight' in key or 'v_proj.weight' in key:
                # Get layer number from key
                layer_num = int(key.split('.')[1])
                base_key = key.replace('k_proj', 'k_proj_a').replace('v_proj', 'v_proj_a')
                
                # Reshape weight for model parallel sharding
                weight = tensor.view(args.kv_lora_rank, args.dim)  # [256, 1536]
                
                # Split into 8 shards
                mp_size = 8
                shard_size = args.kv_lora_rank // mp_size  # 32
                weight = weight.view(mp_size, shard_size, args.dim)  # [8, 32, 1536]
                
                # Add each shard to state dict
                for i in range(mp_size):
                    shard_key = f"{base_key[:-7]}.{i}.weight"  # Replace .weight with shard index
                    state_dict[shard_key] = weight[i].contiguous()
                
                # Add proj_b weights (shared across shards)
                proj_b_key = key.replace('k_proj', 'k_proj_b').replace('v_proj', 'v_proj_b')
                state_dict[proj_b_key] = torch.eye(
                    args.dim,
                    shard_size,
                    device=tensor.device,
                    dtype=tensor.dtype
                )
            # Handle key/value projection biases
            elif 'k_proj.bias' in key or 'v_proj.bias' in key:
                # Get layer number from key
                layer_num = int(key.split('.')[1])
                base_key = key.replace('k_proj', 'k_proj_a').replace('v_proj', 'v_proj_a')
                
                # Split bias into shards
                mp_size = 8
                shard_size = args.kv_lora_rank // mp_size  # 32
                bias = tensor.view(mp_size, shard_size)  # [8, 32]
                
                # Add each shard to state dict
                for i in range(mp_size):
                    bias_key = f"{base_key[:-5]}.{i}.bias"  # Replace .bias with shard index
                    state_dict[bias_key] = bias[i].contiguous()
            else:
                state_dict[key] = tensor
        
        # Load state dict
        self.model.load_state_dict(state_dict)
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