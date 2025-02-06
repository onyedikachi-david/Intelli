import os
import json
import mmap
import requests
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
import torch
from safetensors.torch import safe_open
from huggingface_hub import hf_hub_download

from .tokenizer import DeepSeekTokenizer
from .kernel import act_quant, weight_dequant, fp8_gemm


class TensorType(Enum):
    F32 = 0
    F16 = 1
    Q8_0 = 2  # 8-bit quantization
    Q4_0 = 3  # 4-bit quantization
    Q4_1 = 4  # 4-bit quantization with different grouping


@dataclass
class TensorInfo:
    """Information about a tensor in the model."""
    name: str
    shape: Tuple[int, ...]
    dtype: TensorType
    offset: int
    file_idx: int


class ModelMapping:
    """Memory mapping for model files."""
    def __init__(self, path: str, prefetch: bool = True):
        self.path = path
        self.file = open(path, 'rb')
        self.size = os.path.getsize(path)
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        if prefetch:
            # Prefetch into page cache
            pagesize = mmap.PAGESIZE
            for offset in range(0, self.size, pagesize):
                self.mm[offset:offset + pagesize]
    
    def __del__(self):
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'file'):
            self.file.close()


class ModelLoader:
    """Enhanced model loader with llama.cpp-style optimizations."""
    
    REPO_ID = "deepseek-ai/deepseek-v3"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/deepseek")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.mappings: Dict[str, ModelMapping] = {}
        self.tensor_info: Dict[str, TensorInfo] = {}
        self.size_data = 0
        self.size_done = 0
        self.block_size = 128  # For quantization
        self.use_mmap = True  # Default to using memory mapping
        self.prefetch = True  # Default to prefetching
        self.quantize = False  # Default to no quantization
        self.dtype = torch.bfloat16  # Default dtype
    
    def _download_file(self, url: str, local_path: str) -> None:
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as f, tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    def download_from_hf(self, model_id: str, revision: str = "main") -> str:
        """
        Download model files from HuggingFace using the Hub API.
        
        Args:
            model_id: HuggingFace model ID
            revision: Model revision/tag
            
        Returns:
            Path to downloaded model directory
        """
        # Ensure correct casing for model ID
        if "deepseek" in model_id.lower():
            parts = model_id.split("/")
            if len(parts) == 2:
                # Convert model name part to title case
                model_name_parts = parts[1].split("-")
                model_name_parts = [p.title() for p in model_name_parts]
                parts[1] = "-".join(model_name_parts)
                model_id = "/".join(parts)
        
        model_dir = os.path.join(self.cache_dir, model_id.replace('/', '_'))
        os.makedirs(model_dir, exist_ok=True)
        
        # Files to download
        files = [
            "config.json",
            "tokenizer_config.json",
            "model.safetensors"
        ]
        
        # Try to download tokenizer files from various locations
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer.model",
            "sentencepiece.bpe.model"
        ]
        
        # First download the main files
        for filename in files:
            local_path = os.path.join(model_dir, filename)
            try:
                # Use HF Hub to download files
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    resume_download=True
                )
                
                # If the file exists in a different location, copy it to our model directory
                if os.path.exists(downloaded_path) and downloaded_path != local_path:
                    import shutil
                    shutil.copy2(downloaded_path, local_path)
                    
            except Exception as e:
                print(f"Warning: Failed to download {filename}: {str(e)}")
                continue
        
        # Then try to download tokenizer files
        tokenizer_found = False
        for filename in tokenizer_files:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    resume_download=True
                )
                
                # If we found a tokenizer file, copy it to both tokenizer.json and tokenizer.model
                if os.path.exists(downloaded_path):
                    import shutil
                    if filename.endswith('.json'):
                        shutil.copy2(downloaded_path, os.path.join(model_dir, "tokenizer.json"))
                    else:
                        shutil.copy2(downloaded_path, os.path.join(model_dir, "tokenizer.model"))
                    tokenizer_found = True
                    break
                    
            except Exception as e:
                print(f"Warning: Failed to download {filename}: {str(e)}")
                continue
        
        if not tokenizer_found:
            print("Warning: Could not find any tokenizer files, will attempt to create from scratch")
        
        # Create index file if needed
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            index = {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model": "model.safetensors"
                }
            }
            with open(index_path, 'w') as f:
                json.dump(index, f)
        
        return model_dir
    
    def _init_mappings(self, model_path: str) -> None:
        """Initialize memory mappings for model files."""
        self.model_path = model_path  # Store model path for reference
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        
        # Create or update index file
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {}
        }
        
        # Find all safetensors files in the directory
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                shard_name = os.path.basename(file)
                # Add to weight map if not already present
                if shard_name not in index["weight_map"].values():
                    index["weight_map"][shard_name.replace(".safetensors", "")] = shard_name
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(index, f)
        
        # Initialize safetensors header
        self.safetensors_header = {}
        
        # Create memory mappings for each shard
        for shard in set(index["weight_map"].values()):
            shard_path = os.path.join(model_path, shard)
            if os.path.exists(shard_path):
                self.mappings[shard] = ModelMapping(shard_path, self.prefetch)
                # Load safetensors header for this shard
                with safe_open(shard_path, framework="pt") as f:
                    metadata = f.metadata()
                    for tensor_name in f.keys():
                        tensor = f.get_tensor(tensor_name)
                        # Get tensor info from metadata
                        tensor_info = metadata.get(tensor_name, {})
                        self.safetensors_header[tensor_name] = {
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                            "shape": tuple(tensor.shape),
                            "data_offsets": tensor_info.get("data_offsets", [0]),
                            "shard": shard
                        }
                        
                        # Store tensor info
                        self.tensor_info[tensor_name] = TensorInfo(
                            name=tensor_name,
                            shape=tuple(tensor.shape),
                            dtype=self._get_tensor_type(tensor.dtype),
                            offset=tensor_info.get("data_offsets", [0])[0],
                            file_idx=list(index["weight_map"].values()).index(shard)
                        )
                        self.size_data += np.prod(tensor.shape) * self._get_dtype_size(tensor.dtype)
    
    def _get_tensor_type(self, dtype: torch.dtype) -> TensorType:
        """Map PyTorch dtype to TensorType."""
        if dtype == torch.float32:
            return TensorType.F32
        elif dtype == torch.float16:
            return TensorType.F16
        elif dtype == torch.int8:
            return TensorType.Q8_0
        elif dtype == torch.quint4x2:
            return TensorType.Q4_0
        else:
            return TensorType.F32
    
    def _get_dtype_size(self, dtype: torch.dtype) -> int:
        """Get size in bytes for dtype."""
        if dtype == torch.float32:
            return 4
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            return 2
        elif dtype == torch.int8:
            return 1
        elif dtype == torch.quint4x2:
            return 0.5
        else:
            return 4
    
    def _load_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """Load a tensor from the safetensors file."""
        info = self.safetensors_header[name]
        
        # Get tensor info
        dtype = self._dtype_from_str(info["dtype"])
        original_shape = info["shape"]
        shape = original_shape  # Default to original shape
        
        # Load tensor data first
        tensor = self._load_tensor_data(name, info, dtype)
        actual_size = tensor.numel()
        
        # Get model configuration based on checkpoint dimensions
        # First, find a layer's input layernorm to determine hidden_dim
        hidden_dim = None
        for key in self.safetensors_header:
            if "input_layernorm.weight" in key:
                hidden_dim = self.safetensors_header[key]["shape"][0]
                break
        if hidden_dim is None:
            # Try to get from embeddings
            for key in self.safetensors_header:
                if key.endswith("embed_tokens.weight"):
                    hidden_dim = self.safetensors_header[key]["shape"][1]
                    break
        if hidden_dim is None:
            hidden_dim = original_shape[1] if len(original_shape) > 1 else original_shape[0]
        
        # Find MLP dimensions from gate_proj or up_proj
        intermediate_dim = None
        for key in self.safetensors_header:
            if "mlp.gate_proj.weight" in key:
                intermediate_dim = self.safetensors_header[key]["shape"][0]
                break
            elif "mlp.up_proj.weight" in key:
                intermediate_dim = self.safetensors_header[key]["shape"][0]
                break
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 4  # Common ratio in transformer models
        
        # Calculate attention dimensions from checkpoint
        head_dim = None
        num_key_value_heads = None
        for key in self.safetensors_header:
            if "self_attn.v_proj_a.0.weight" in key:
                head_dim = self.safetensors_header[key]["shape"][0]
                # Count number of v_proj_a shards to determine KV heads
                num_key_value_heads = len([k for k in self.safetensors_header if "v_proj_a" in k and k.endswith(".weight")])
                break
        if head_dim is None or num_key_value_heads is None:
            # Try to determine from o_proj dimensions
            for key in self.safetensors_header:
                if "self_attn.o_proj.weight" in key:
                    o_proj_shape = self.safetensors_header[key]["shape"]
                    head_dim = o_proj_shape[0] // 48  # Common ratio
                    num_key_value_heads = o_proj_shape[0] // head_dim
                    break
        if head_dim is None:
            head_dim = 32  # Fallback
            num_key_value_heads = hidden_dim // head_dim
        
        num_attention_heads = num_key_value_heads
        
        # Map tensor names to remove "model." prefix if present
        clean_name = name.replace("model.", "")
        
        # Handle attention projection layers
        if "self_attn" in clean_name:
            if "proj_a" in clean_name:
                # Handle sharded key/value projection weights
                if any(x in clean_name for x in ["k_proj_a", "v_proj_a"]):
                    # Keep original shape for these layers
                    shape = original_shape
                else:
                    # Query projection - keep original shape
                    shape = original_shape
            elif "proj_b" in clean_name:
                # Handle key/value projection second matrix - keep original shape
                shape = original_shape
            elif "o_proj" in clean_name:
                # Output projection - keep original shape
                shape = original_shape
        # Handle embedding and lm_head
        elif clean_name in ["embed_tokens.weight", "lm_head.weight"]:
            # Keep original shapes for these layers
            shape = original_shape
        # Handle MLP layers
        elif "mlp." in clean_name:
            # Keep original shapes for MLP layers
            shape = original_shape
        # Handle norm layers
        elif any(x in clean_name for x in ["norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]):
            # Keep original shapes for norm layers
            shape = original_shape
        
        # Reshape and move to device
        try:
            tensor = tensor.reshape(shape)
        except ValueError as e:
            print(f"Error reshaping tensor {name}")
            print(f"Tensor size: {tensor.size()}")
            print(f"Target shape: {shape}")
            print(f"Original shape: {original_shape}")
            raise e
            
        if device != "cpu":
            tensor = tensor.to(device)
            
        return tensor
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor using block-wise quantization."""
        if tensor.element_size() > 2:  # Only quantize fp32/fp16 tensors
            tensor, scale = act_quant(tensor, self.block_size)
            tensor.scale = scale  # Store scale for dequantization
        return tensor
    
    def _load_tensor_data(self, name: str, info: Dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
        """Load raw tensor data from file."""
        shard = self.mappings[info["shard"]]
        offset = info["data_offsets"][0]
        
        # Calculate tensor size
        shape = info["shape"]
        size = np.prod(shape) * self._get_dtype_size(dtype)
        
        # Read data from memory mapping
        data = np.frombuffer(
            shard.mm[offset:offset + size],
            dtype=np.float32 if dtype == torch.float32 else np.float16
        )
        
        # Convert to torch tensor
        tensor = torch.from_numpy(data)
        
        # Convert dtype if needed
        if dtype != tensor.dtype:
            tensor = tensor.to(dtype)
        
        return tensor
    
    def _dtype_from_str(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "uint8": torch.uint8
        }
        return dtype_map.get(dtype_str, torch.float32)
    
    def load_model(self, model_path: str, device: str = "cuda",
                  quantize: bool = False, dtype: Optional[torch.dtype] = None,
                  progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Load model weights with memory optimizations.
        
        Args:
            model_path: Path to model directory
            device: Device to load tensors to
            quantize: Whether to use quantization
            dtype: Data type for tensors
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing model components (config, tokenizer, weights)
        """
        self.quantize = quantize
        self.dtype = dtype or self.dtype
        
        # Load config with default values
        config = {
            "model_type": "deepseek",
            "vocab_size": 151936,
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "intermediate_size": 8960,
            "max_position_embeddings": 8192,
            "max_sequence_length": 8192,
            "use_cache": True,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": True,
            "dtype": "bf16"
        }
        
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    loaded_config = json.load(f)
                    config.update(loaded_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load config.json: {str(e)}. Using default values.")
            
        # Load tokenizer
        try:
            tokenizer = DeepSeekTokenizer(model_path)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {str(e)}. Using default tokenizer.")
            tokenizer = DeepSeekTokenizer(None)  # Use default tokenizer
        
        # Initialize memory mappings
        self._init_mappings(model_path)
        
        # Load tensors
        tensors = {}
        for name in tqdm(self.tensor_info.keys(), desc="Loading tensors"):
            tensor = self._load_tensor(name, device)
            tensors[name] = tensor
            
            self.size_done += np.prod(tensor.shape) * tensor.element_size()
            if progress_callback:
                progress = self.size_done / self.size_data
                progress_callback(progress)
        
        return {
            "config": config,
            "tokenizer": tokenizer,
            "weights": tensors,
            "device": device
        }


class LazyTensor:
    """Memory-efficient tensor that loads data on demand."""
    
    def __init__(self, loader: ModelLoader, name: str, device: str, dtype: Optional[torch.dtype] = None):
        self.loader = loader
        self.name = name
        self.device = device
        self.dtype = dtype
        self._tensor = None
        
        # Get tensor metadata
        info = loader.tensor_info[name]
        self.shape = info.shape
    
    def materialize(self) -> torch.Tensor:
        """Load tensor into memory."""
        if self._tensor is None:
            self._tensor = self.loader._load_tensor(self.name, self.device)
        return self._tensor
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'file'):
            self.file.close() 