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

from .tokenizer import DeepSeekTokenizer


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
        Download model files from HuggingFace.
        
        Args:
            model_id: HuggingFace model ID
            revision: Model revision/tag
            
        Returns:
            Path to downloaded model directory
        """
        model_dir = os.path.join(self.cache_dir, model_id.replace('/', '_'))
        os.makedirs(model_dir, exist_ok=True)
        
        # Download model files
        files = ["config.json", "tokenizer.model", "tokenizer_config.json", "model.safetensors"]
        base_url = f"https://huggingface.co/{model_id}/resolve/{revision}"
        
        for filename in files:
            local_path = os.path.join(model_dir, filename)
            if not os.path.exists(local_path):
                url = f"{base_url}/{filename}"
                try:
                    self._download_file(url, local_path)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        print(f"Warning: {filename} not found, skipping...")
                        continue
                    raise
        
        # Create a simple index file if not downloaded
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
    
    def _init_mappings(self, model_path: str, prefetch: bool = True) -> None:
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
                self.mappings[shard] = ModelMapping(shard_path, prefetch)
            
        # Build tensor info
        for name, shard in index["weight_map"].items():
            shard_path = os.path.join(model_path, shard)
            if os.path.exists(shard_path):
                with safe_open(shard_path, framework="pt") as f:
                    for tensor_name in f.keys():
                        tensor = f.get_tensor(tensor_name)
                        self.tensor_info[tensor_name] = TensorInfo(
                            name=tensor_name,
                            shape=tuple(tensor.shape),
                            dtype=self._get_tensor_type(tensor.dtype),
                            offset=f.get_tensor_info(tensor_name)["data_offsets"][0],
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
        else:
            return TensorType.F32
    
    def _get_dtype_size(self, dtype: torch.dtype) -> int:
        """Get size in bytes for dtype."""
        if dtype == torch.float32:
            return 4
        elif dtype == torch.float16:
            return 2
        elif dtype == torch.int8:
            return 1
        else:
            return 4
    
    def _load_tensor(self, name: str, device: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Load a single tensor from memory mapping."""
        info = self.tensor_info[name]
        shard = list(self.mappings.values())[info.file_idx]
        
        # Get tensor data from memory mapping
        tensor_size = np.prod(info.shape) * self._get_dtype_size(dtype or torch.float32)
        tensor_data = memoryview(shard.mm[info.offset:info.offset + tensor_size])
        
        # Convert to tensor
        tensor = torch.frombuffer(tensor_data, dtype=dtype or torch.float32)
        tensor = tensor.reshape(info.shape)
        
        # Quantize if needed
        if info.dtype in [TensorType.Q8_0, TensorType.Q4_0, TensorType.Q4_1]:
            tensor = self._quantize_tensor(tensor, info.dtype)
        
        return tensor.to(device=device)
    
    def _quantize_tensor(self, tensor: torch.Tensor, qtype: TensorType) -> torch.Tensor:
        """Quantize tensor to specified type."""
        if qtype == TensorType.Q8_0:
            # Simple 8-bit quantization
            scale = tensor.abs().max() / 127
            return torch.round(tensor / scale).to(torch.int8) * scale
        elif qtype in [TensorType.Q4_0, TensorType.Q4_1]:
            # 4-bit quantization (simplified)
            scale = tensor.abs().max() / 7
            return torch.round(tensor / scale).clamp(-7, 7).to(torch.int8) * scale
        return tensor
    
    def load_model(self, model_path: str, device: str = "cuda",
                  quantize: bool = False, dtype: Optional[torch.dtype] = None,
                  progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Load model with memory optimizations.
        
        Args:
            model_path: Path to model directory
            device: Device to load model on
            quantize: Whether to quantize the model
            dtype: Data type for model weights
            progress_callback: Optional callback for loading progress
            
        Returns:
            Dictionary containing model components
        """
        # Initialize mappings
        self._init_mappings(model_path, prefetch=True)
        
        # Load config and tokenizer
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
        tokenizer = DeepSeekTokenizer(model_path)
        
        # Load weights with progress tracking
        weights = {}
        for name in tqdm(self.tensor_info.keys(), desc="Loading weights"):
            weights[name] = LazyTensor(
                loader=self,
                name=name,
                device=device,
                dtype=dtype
            )
            self.size_done += np.prod(self.tensor_info[name].shape) * self._get_dtype_size(dtype or torch.float32)
            
            if progress_callback:
                progress = self.size_done / self.size_data
                if not progress_callback(progress):
                    break
        
        return {
            "config": config,
            "tokenizer": tokenizer,
            "weights": weights,
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
            self._tensor = self.loader._load_tensor(self.name, self.device, self.dtype)
        return self._tensor
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'file'):
            self.file.close() 