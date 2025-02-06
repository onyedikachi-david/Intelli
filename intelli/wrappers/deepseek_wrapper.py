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
        
        # Default generation parameters - balanced for general use
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 40
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
        
        # Initialize model with config from checkpoint
        self.config = model_data["config"]
        self.tokenizer = model_data["tokenizer"]
        
        # Create model args from config
        model_args = {
            'dim': self.config["hidden_size"],
            'n_layers': self.config["num_hidden_layers"],
            'n_heads': self.config["num_attention_heads"],
            'vocab_size': self.config["vocab_size"],
            'max_seq_len': self.config["max_sequence_length"],
            'max_batch_size': 32,
            'inter_dim': self.config["intermediate_size"],
            'dtype': self.dtype,
            'rope_theta': 10000.0,
            'rope_factor': 40,
            'beta_fast': 32,
            'beta_slow': 1,
            'mscale': 1.0
        }
        
        # Initialize model with args from checkpoint
        args = ModelArgs(**model_args)
        self.model = Transformer(args).to(self.device)
        
        # Load state dict
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
                
                # Get dimensions from tensor
                out_dim, in_dim = tensor.shape
                
                # Split into 8 shards
                mp_size = 8
                shard_size = out_dim // mp_size
                weight = tensor.view(mp_size, shard_size, in_dim)  # [8, shard_size, in_dim]
                
                # Add each shard to state dict
                for i in range(mp_size):
                    shard_key = f"{base_key[:-7]}.{i}.weight"  # Replace .weight with shard index
                    state_dict[shard_key] = weight[i].contiguous()
                
                # Add proj_b weights (identity matrix for each shard)
                proj_b_key = key.replace('k_proj', 'k_proj_b').replace('v_proj', 'v_proj_b')
                state_dict[proj_b_key] = torch.eye(
                    in_dim,
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
                shard_size = tensor.size(0) // mp_size
                bias = tensor.view(mp_size, shard_size)  # [8, shard_size]
                
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
        
        try:
            # Format as chat messages
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Get input tokens
            input_ids = self.tokenizer.apply_chat_template(messages)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Track generated tokens for repetition penalty
            generated = []
            response_text = ""
            consecutive_spaces = 0
            
            # Get vocabulary sizes
            config_vocab_size = self.config["vocab_size"]
            tokenizer_vocab_size = self.tokenizer.sp_model.get_piece_size()
            # Use tokenizer vocab size since that's what we can actually decode
            vocab_size = tokenizer_vocab_size
            print(f"Config vocab size: {config_vocab_size}")
            print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
            print(f"Using vocab size: {vocab_size}")
            
            # Get special token IDs
            special_tokens = {
                self.tokenizer.pad_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.user_token_id,
                self.tokenizer.assistant_token_id,
                self.tokenizer.system_token_id,
                self.tokenizer.sp_model.unk_id()
            }
            print(f"Special token IDs: {special_tokens}")
            
            # Generate tokens
            print(f"\nGeneration started (max_length={self.max_length})...")
            
            # Initial forward pass
            with torch.no_grad():
                # Debug input shape
                print(f"Input shape: {input_ids.shape}")
                
                outputs = self.model(input_ids)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Print shape info for debugging
                print(f"Model output shape: {logits.shape}")
                print(f"Output dtype: {logits.dtype}")
                
                # Convert to float32 for better numerical stability
                logits = logits.to(torch.float32)
                
                # Get last token logits based on shape
                if len(logits.shape) == 3:
                    # Get full logits first
                    next_token_logits = logits[0, -1].clone()
                    # Create a mask for valid token IDs
                    valid_tokens_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    valid_tokens_mask[:vocab_size] = True
                    # Set invalid token logits to large negative value
                    next_token_logits = torch.where(
                        valid_tokens_mask,
                        next_token_logits,
                        torch.full_like(next_token_logits, float('-inf'))
                    )
                elif len(logits.shape) == 2:
                    # Same for 2D logits
                    next_token_logits = logits[-1].clone()
                    valid_tokens_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    valid_tokens_mask[:vocab_size] = True
                    next_token_logits = torch.where(
                        valid_tokens_mask,
                        next_token_logits,
                        torch.full_like(next_token_logits, float('-inf'))
                    )
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
                
                # Replace NaN/Inf values with large negative numbers only for valid tokens
                next_token_logits = torch.where(
                    torch.isnan(next_token_logits) | torch.isinf(next_token_logits),
                    torch.full_like(next_token_logits, -1e4),
                    next_token_logits
                )
                
                # Check for inf/nan values after replacement
                print(f"Has inf values after fix: {torch.isinf(next_token_logits).any().item()}")
                print(f"Has nan values after fix: {torch.isnan(next_token_logits).any().item()}")
                
                # Print initial logits stats for debugging
                valid_logits = next_token_logits[:vocab_size]
                print(f"Initial logits - min: {valid_logits.min().item():.2f}, max: {valid_logits.max().item():.2f}, mean: {valid_logits.mean().item():.2f}")
                
                for i in range(self.max_length):
                    # Apply repetition penalty only to valid tokens
                    if len(generated) > 0:
                        for token in generated:
                            if token < vocab_size:  # Only apply to valid token IDs
                                if next_token_logits[token] > 0:
                                    next_token_logits[token] /= self.repetition_penalty
                                else:
                                    next_token_logits[token] *= self.repetition_penalty
                    
                    # Apply temperature scaling
                    if self.temperature > 0:
                        scaled_logits = next_token_logits / max(self.temperature, 1e-6)
                    else:
                        scaled_logits = next_token_logits
                    
                    # Apply top-k filtering only to valid tokens
                    if self.top_k > 0:
                        # Get top-k only from valid tokens
                        valid_logits = scaled_logits[:vocab_size]
                        values, _ = torch.topk(valid_logits, min(self.top_k, vocab_size))
                        min_value = values[-1]
                        scaled_logits = torch.where(
                            scaled_logits < min_value,
                            torch.full_like(scaled_logits, float('-inf')),
                            scaled_logits
                        )
                    
                    # Apply top-p (nucleus) filtering only to valid tokens
                    if self.top_p < 1.0:
                        # Sort only valid tokens
                        valid_logits = scaled_logits[:vocab_size]
                        sorted_logits, sorted_indices = torch.sort(valid_logits, descending=True)
                        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > self.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Map back to original indices
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        scaled_logits[:vocab_size][indices_to_remove] = float('-inf')
                    
                    # Ensure finite values for softmax (only for valid tokens)
                    valid_logits = scaled_logits[:vocab_size]
                    max_logit = valid_logits.max()
                    valid_logits = valid_logits - max_logit
                    scaled_logits[:vocab_size] = valid_logits
                    
                    # Handle any remaining inf values for valid tokens
                    scaled_logits[:vocab_size] = torch.where(
                        torch.isinf(valid_logits),
                        torch.full_like(valid_logits, -1e4),
                        valid_logits
                    )
                    
                    # Apply softmax only to valid tokens
                    valid_logits = scaled_logits[:vocab_size]
                    exp_logits = torch.exp(valid_logits)
                    probs = torch.zeros_like(scaled_logits)
                    probs[:vocab_size] = exp_logits / (exp_logits.sum() + 1e-10)
                    
                    # Debug probability distribution
                    if i == 0:
                        valid_probs = probs[:vocab_size]
                        print(f"Probability sum: {valid_probs.sum().item():.6f}")
                        print(f"Max probability: {valid_probs.max().item():.6f}")
                        print(f"Has valid distribution: {(valid_probs >= 0).all().item() and (valid_probs <= 1).all().item()}")
                        top_probs, top_indices = valid_probs.topk(5)
                        print(f"Top 5 probabilities: {top_probs.tolist()}")
                        print(f"Top 5 token IDs: {top_indices.tolist()}")
                        # Print actual tokens for debugging
                        print("Top 5 tokens:", [self.tokenizer.decode([idx.item()]) for idx in top_indices])
                    
                    # Ensure valid probabilities
                    if torch.isnan(probs).any() or (probs[:vocab_size].sum() - 1.0).abs() > 1e-3:
                        print("\nWarning: Invalid probabilities detected, falling back to greedy selection")
                        # Use argmax only on valid tokens
                        next_token = torch.argmax(scaled_logits[:vocab_size]).reshape(1)
                    else:
                        # Sample only from valid tokens
                        next_token = torch.multinomial(probs[:vocab_size], num_samples=1)
                    
                    # Validate token ID
                    token_id = next_token.item()
                    if token_id >= vocab_size or token_id < 0:
                        print(f"\nWarning: Token ID {token_id} out of range, using UNK token")
                        token_id = self.tokenizer.sp_model.unk_id()
                        next_token = torch.tensor([token_id], device=self.device)
                    
                    # Try decoding the token to validate it
                    try:
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        if not token_text:  # If empty string returned
                            print(f"\nWarning: Empty token text for ID {token_id}, using UNK token")
                            token_id = self.tokenizer.sp_model.unk_id()
                            next_token = torch.tensor([token_id], device=self.device)
                            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    except Exception as e:
                        print(f"\nWarning: Failed to decode token {token_id}, using UNK token")
                        token_id = self.tokenizer.sp_model.unk_id()
                        next_token = torch.tensor([token_id], device=self.device)
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    
                    # Add the chosen token to the sequence
                    generated.append(token_id)
                    
                    # Reshape next_token to match input_ids dimensions [batch_size, seq_len]
                    next_token = next_token.unsqueeze(0)  # Add batch dimension
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    # Get next token's logits
                    outputs = self.model(input_ids)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Convert to float32 for better numerical stability
                    logits = logits.to(torch.float32)
                    
                    # Get next token logits based on shape
                    if len(logits.shape) == 3:
                        # Get full logits first
                        next_token_logits = logits[0, -1].clone()
                        # Create a mask for valid token IDs
                        valid_tokens_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                        valid_tokens_mask[:vocab_size] = True
                        # Set invalid token logits to large negative value
                        next_token_logits = torch.where(
                            valid_tokens_mask,
                            next_token_logits,
                            torch.full_like(next_token_logits, float('-inf'))
                        )
                    elif len(logits.shape) == 2:
                        # Same for 2D logits
                        next_token_logits = logits[-1].clone()
                        valid_tokens_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                        valid_tokens_mask[:vocab_size] = True
                        next_token_logits = torch.where(
                            valid_tokens_mask,
                            next_token_logits,
                            torch.full_like(next_token_logits, float('-inf'))
                        )
                    
                    # Replace NaN/Inf values only for valid tokens
                    next_token_logits[:vocab_size] = torch.where(
                        torch.isnan(next_token_logits[:vocab_size]) | torch.isinf(next_token_logits[:vocab_size]),
                        torch.full_like(next_token_logits[:vocab_size], -1e4),
                        next_token_logits[:vocab_size]
                    )
                    
                    if token_text:
                        response_text += token_text
                        # Update consecutive spaces counter
                        if token_text.isspace():
                            consecutive_spaces += 1
                        else:
                            consecutive_spaces = 0
                        # Print progress
                        print(f"\rGenerated ({i+1} tokens): {response_text}", end="", flush=True)
                    
                    # Check for stop conditions
                    if token_id in [self.tokenizer.eos_token_id, self.tokenizer.user_token_id]:
                        print("\nGeneration complete: End token reached")
                        break
                    elif consecutive_spaces >= 5:  # Reduced threshold for consecutive spaces
                        print("\nGeneration complete: Multiple spaces detected")
                        break
                    elif len(response_text) > 0 and not response_text[-1].strip():
                        # Check if we've hit a natural stopping point (sentence end + space)
                        last_char = response_text.rstrip()[-1] if response_text.rstrip() else ""
                        if last_char in ".!?" and i > 20:
                            print("\nGeneration complete: Natural end point reached")
                            break
            
            print("\n")  # New line after generation
            
            # Restore original parameters if they were temporarily overridden
            if kwargs:
                self.update_params(**temp_params)
            
            return response_text.strip()
            
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Alias for generate method."""
        return self.generate(prompt, **kwargs)

    def update_model_params(self, model_params: Dict[str, Any]):
        """Update model parameters."""
        self.model_params = model_params 