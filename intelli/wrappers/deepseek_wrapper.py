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
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        loader = ModelLoader()
        
        # Download model files if needed
        model_path = loader.download_from_hf(self.model_variant)
        
        # Load model with optimizations
        model_data = loader.load_model(
            model_path,
            device="cpu",  # Load on CPU first
            quantize=False,  # Disable quantization during loading
            dtype=self.dtype  # Use target dtype for loading
        )
        
        # Initialize model with config from checkpoint
        self.config = model_data["config"]
        self.tokenizer = model_data["tokenizer"]
        
        # Get vocabulary size from config or tokenizer
        if "vocab_size" in self.config:
            vocab_size = self.config["vocab_size"]
        else:
            vocab_size = self.tokenizer.sp_model.get_piece_size()
            
        print(f"\nVocabulary information:")
        print(f"Config vocab size: {self.config.get('vocab_size', 'Not specified')}")
        print(f"Tokenizer vocab size: {vocab_size}")
        
        # Create model args from config
        model_args = {
            'dim': self.config["hidden_size"],
            'n_layers': self.config["num_hidden_layers"],
            'n_heads': self.config["num_attention_heads"],
            'vocab_size': vocab_size,
            'max_seq_len': self.config["max_sequence_length"],
            'max_batch_size': 32,
            'inter_dim': self.config["intermediate_size"],
            'dtype': self.dtype,  # Use target dtype
            'rope_theta': 10000.0,
            'rope_factor': 40,
            'beta_fast': 32,
            'beta_slow': 1,
            'mscale': 1.0
        }
        
        # Initialize model with args from checkpoint
        args = ModelArgs(**model_args)
        self.model = Transformer(args)
        
        # Load state dict
        state_dict = {}
        for key, tensor in model_data["weights"].items():
            if isinstance(tensor, LazyTensor):
                tensor = tensor.materialize()
            # Remove "model." prefix from key
            if key.startswith("model."):
                key = key[6:]  # Remove "model." prefix
            
            # Convert tensor to target dtype and normalize if needed
            tensor = tensor.to(self.dtype)
            if tensor.dim() > 0:  # Skip scalars
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    # Replace NaN/Inf with zeros
                    tensor = torch.where(
                        torch.isnan(tensor) | torch.isinf(tensor),
                        torch.zeros_like(tensor),
                        tensor
                    )
                # Normalize large weights
                if tensor.abs().max() > 100:
                    tensor = tensor * (100 / tensor.abs().max())
            
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
                    dtype=self.dtype,
                    device="cpu"
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
            # Handle embedding and output layers
            elif key in ['embed_tokens.weight', 'lm_head.weight']:
                # Keep original vocab size for these layers
                if tensor.size(0) != vocab_size:
                    if tensor.size(0) > vocab_size:
                        # Truncate if larger
                        tensor = tensor[:vocab_size]
                    else:
                        # Pad if smaller
                        pad_size = vocab_size - tensor.size(0)
                        tensor = torch.cat([
                            tensor,
                            torch.zeros(pad_size, tensor.size(1), dtype=tensor.dtype)
                        ])
                # Normalize embedding weights
                if tensor.dim() > 0:
                    tensor = tensor / max(tensor.norm(dim=-1).max().item(), 1e-3)
                state_dict[key] = tensor
            else:
                state_dict[key] = tensor
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
        
        # Move model to device
        self.model = self.model.to(self.device)
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Verify model loaded correctly
        print("\nModel initialization:")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print(f"Vocab size: {self.model.args.vocab_size}")
        print(f"Hidden size: {self.model.args.dim}")
        print(f"Num layers: {self.model.args.n_layers}")
        print(f"Num heads: {self.model.args.n_heads}")
        
        # Test forward pass
        with torch.no_grad():
            test_input = torch.ones((1, 1), dtype=torch.long, device=self.device)
            try:
                test_output = self.model(test_input)
                print("Test forward pass successful")
                print(f"Output shape: {test_output.shape}")
                print(f"Output dtype: {test_output.dtype}")
                print(f"Output device: {test_output.device}")
                print(f"Output has NaN: {torch.isnan(test_output).any().item()}")
                print(f"Output has Inf: {torch.isinf(test_output).any().item()}")
                print(f"Output stats - min: {test_output.min().item():.2f}, max: {test_output.max().item():.2f}, mean: {test_output.mean().item():.2f}")
                
                # Additional checks for numerical stability
                if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                    print("\nWarning: Found NaN/Inf in test output")
                    print("Layer-wise parameter stats:")
                    for name, param in self.model.named_parameters():
                        if param.dim() > 0:  # Skip scalars
                            print(f"{name}:")
                            print(f"  min: {param.min().item():.2f}")
                            print(f"  max: {param.max().item():.2f}")
                            print(f"  mean: {param.mean().item():.2f}")
                            print(f"  std: {param.std().item():.2f}")
                            print(f"  has NaN: {torch.isnan(param).any().item()}")
                            print(f"  has Inf: {torch.isinf(param).any().item()}")
                
            except Exception as e:
                print(f"Test forward pass failed: {str(e)}")
                raise

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
        """Generate text from prompt."""
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
            tokenizer_vocab_size = self.tokenizer.sp_model.get_piece_size()
            model_vocab_size = self.model.args.vocab_size
            print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
            print(f"Model vocab size: {model_vocab_size}")
            
            # Use minimum of tokenizer and model vocab size
            vocab_size = min(tokenizer_vocab_size, model_vocab_size)
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
                
                # Forward pass with gradient checkpointing disabled
                self.model.gradient_checkpointing = False
                outputs = self.model(input_ids)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Print shape info for debugging
                print(f"Model output shape: {logits.shape}")
                print(f"Output dtype: {logits.dtype}")
                print(f"Output device: {logits.device}")
                
                # Get last token logits based on shape
                if len(logits.shape) == 3:
                    next_token_logits = logits[0, -1, :vocab_size].clone()  # Only keep valid vocab
                elif len(logits.shape) == 2:
                    next_token_logits = logits[-1, :vocab_size].clone()  # Only keep valid vocab
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
                
                # Keep logits in model dtype and handle NaN/Inf
                next_token_logits = next_token_logits.to(self.dtype)
                next_token_logits = torch.where(
                    torch.isnan(next_token_logits) | torch.isinf(next_token_logits),
                    torch.zeros_like(next_token_logits),
                    next_token_logits
                )
                
                # Clip extreme values
                max_value = 100
                next_token_logits = torch.clamp(next_token_logits, min=-max_value, max=max_value)
                
                # Print raw logits stats
                print(f"Raw logits - min: {next_token_logits.min().item():.2f}, max: {next_token_logits.max().item():.2f}, mean: {next_token_logits.mean().item():.2f}")
                
                # Normalize logits to prevent overflow
                next_token_logits = next_token_logits - next_token_logits.max()
                
                # Add small epsilon to avoid numerical instability
                eps = torch.finfo(self.dtype).tiny
                next_token_logits = next_token_logits + eps
                
                # Apply temperature scaling first
                if self.temperature > 0:
                    next_token_logits = next_token_logits / max(self.temperature, eps)
                
                # Apply softmax with better numerical stability
                max_logits = next_token_logits.max()
                exp_logits = torch.exp(next_token_logits - max_logits)
                probs = exp_logits / exp_logits.sum()
                
                # Ensure valid probability distribution
                probs = torch.where(
                    torch.isnan(probs) | torch.isinf(probs) | (probs < 0),
                    torch.ones_like(probs) / probs.size(-1),
                    probs
                )
                probs = probs / probs.sum()  # Renormalize
                
                # Debug probability distribution
                print(f"Probability sum: {probs.sum().item():.6f}")
                print(f"Max probability: {probs.max().item():.6f}")
                print(f"Has valid distribution: {(probs >= 0).all().item() and (probs <= 1).all().item()}")
                
                top_probs, top_indices = probs.topk(5)
                print(f"Top 5 probabilities: {top_probs.tolist()}")
                print(f"Top 5 token IDs: {top_indices.tolist()}")
                print("Top 5 tokens:", [self.tokenizer.decode([idx.item()]) for idx in top_indices])
                
                for i in range(self.max_length):
                    # Apply repetition penalty
                    if len(generated) > 0:
                        for token in generated:
                            if token < vocab_size:
                                if next_token_logits[token] > 0:
                                    next_token_logits[token] /= self.repetition_penalty
                                else:
                                    next_token_logits[token] *= self.repetition_penalty
                    
                    # Normalize logits again after repetition penalty
                    next_token_logits = next_token_logits - next_token_logits.max()
                    next_token_logits = next_token_logits + eps
                    
                    # Apply temperature scaling
                    if self.temperature > 0:
                        scaled_logits = next_token_logits / max(self.temperature, eps)
                    else:
                        scaled_logits = next_token_logits
                    
                    # Apply top-k filtering
                    if self.top_k > 0:
                        values, _ = torch.topk(scaled_logits, min(self.top_k, vocab_size))
                        min_value = values[-1]
                        scaled_logits = torch.where(
                            scaled_logits < min_value,
                            torch.full_like(scaled_logits, float('-inf')),
                            scaled_logits
                        )
                    
                    # Apply top-p filtering
                    if self.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > self.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        scaled_logits[indices_to_remove] = float('-inf')
                    
                    # Get probabilities with better numerical stability
                    max_logits = scaled_logits.max()
                    exp_logits = torch.exp(scaled_logits - max_logits)
                    probs = exp_logits / exp_logits.sum()
                    
                    # Ensure valid probability distribution
                    probs = torch.where(
                        torch.isnan(probs) | torch.isinf(probs) | (probs < 0),
                        torch.ones_like(probs) / probs.size(-1),
                        probs
                    )
                    probs = probs / probs.sum()  # Renormalize
                    
                    # Sample next token
                    try:
                        next_token = torch.multinomial(probs, num_samples=1)
                    except RuntimeError as e:
                        print(f"\nError sampling token: {str(e)}")
                        print(f"Probability stats - min: {probs.min().item():.6f}, max: {probs.max().item():.6f}, sum: {probs.sum().item():.6f}")
                        # Fallback to argmax
                        next_token = probs.argmax().unsqueeze(0)
                    
                    token_id = next_token.item()
                    
                    # Validate token ID
                    if token_id >= vocab_size:
                        print(f"\nWarning: Token ID {token_id} out of range, using UNK token")
                        token_id = self.tokenizer.sp_model.unk_id()
                        next_token = torch.tensor([token_id], device=self.device)
                    
                    # Try decoding the token
                    try:
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        if not token_text:
                            print(f"\nWarning: Empty token text for ID {token_id}, using UNK token")
                            token_id = self.tokenizer.sp_model.unk_id()
                            next_token = torch.tensor([token_id], device=self.device)
                            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    except Exception as e:
                        print(f"\nWarning: Failed to decode token {token_id}, using UNK token")
                        token_id = self.tokenizer.sp_model.unk_id()
                        next_token = torch.tensor([token_id], device=self.device)
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    
                    # Add token to sequence
                    generated.append(token_id)
                    next_token = next_token.unsqueeze(0)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    # Update response text
                    if token_text:
                        response_text += token_text
                        if token_text.isspace():
                            consecutive_spaces += 1
                        else:
                            consecutive_spaces = 0
                        print(f"\rGenerated ({i+1} tokens): {response_text}", end="", flush=True)
                    
                    # Get next token logits
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        
                        if len(logits.shape) == 3:
                            next_token_logits = logits[0, -1, :vocab_size].clone()  # Only keep valid vocab
                        elif len(logits.shape) == 2:
                            next_token_logits = logits[-1, :vocab_size].clone()  # Only keep valid vocab
                        
                        # Keep logits in model dtype and handle NaN/Inf
                        next_token_logits = next_token_logits.to(self.dtype)
                        next_token_logits = torch.where(
                            torch.isnan(next_token_logits) | torch.isinf(next_token_logits),
                            torch.zeros_like(next_token_logits),
                            next_token_logits
                        )
                        
                        # Clip extreme values
                        next_token_logits = torch.clamp(next_token_logits, min=-max_value, max=max_value)
                        
                        # Normalize logits
                        next_token_logits = next_token_logits - next_token_logits.max()
                        next_token_logits = next_token_logits + eps
            
            print("\n")
            
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