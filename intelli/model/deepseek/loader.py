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
        
        # Create memory mappings for each shard
        for shard in set(index["weight_map"].values()):
            shard_path = os.path.join(model_path, shard)
            if os.path.exists(shard_path):
                self.mappings[shard] = ModelMapping(shard_path, self.prefetch)
            
        # Build tensor info
        for name, shard in index["weight_map"].items():
            shard_path = os.path.join(model_path, shard)
            if os.path.exists(shard_path):
                with safe_open(shard_path, framework="pt") as f:
                    metadata = f.metadata()
                    for tensor_name in f.keys():
                        tensor = f.get_tensor(tensor_name)
                        # Get tensor offset from metadata
                        tensor_info = metadata.get(tensor_name, {})
                        offset = tensor_info.get("data_offsets", [0])[0] if tensor_info else 0
                        
                        self.tensor_info[tensor_name] = TensorInfo(
                            name=tensor_name,
                            shape=tuple(tensor.shape),
                            dtype=self._get_tensor_type(tensor.dtype),
                            offset=offset,
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
    
    def _load_tensor(self, name: str, device: str = "cuda") -> torch.Tensor:
        """Load a tensor from memory mapping."""
        info = self.tensor_info[name]
        shard = list(self.mappings.values())[info.file_idx]
        
        # Read tensor data from memory mapping
        tensor_size = np.prod(info.shape) * self._get_dtype_size(self.dtype)
        tensor_data = np.frombuffer(
            shard.mm[info.offset:info.offset + tensor_size],
            dtype=np.float32 if info.dtype == TensorType.F32 else np.float16
        ).reshape(info.shape)
        
        # Convert to torch tensor
        tensor = torch.from_numpy(tensor_data).to(device)
        
        # Apply quantization if enabled
        if self.quantize and info.dtype not in [TensorType.Q8_0, TensorType.Q4_0, TensorType.Q4_1]:
            tensor = self._quantize_tensor(tensor)
        
        return tensor
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor using block-wise quantization."""
        if tensor.element_size() > 2:  # Only quantize fp32/fp16 tensors
            tensor, scale = act_quant(tensor, self.block_size)
            tensor.scale = scale  # Store scale for dequantization
        return tensor
    
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
            Dictionary of model tensors
        """
        self.quantize = quantize
        self.dtype = dtype or self.dtype
        
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
        
        return tensors


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