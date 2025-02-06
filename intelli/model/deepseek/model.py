import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .kernel import act_quant, weight_dequant, fp8_gemm


@dataclass
class ModelArgs:
    """Model configuration arguments."""
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 1536  # Hidden dimension matching checkpoint
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    q_lora_rank: int = 0
    kv_lora_rank: int = 256  # LoRA rank matching checkpoint
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.qk_rope_head_dim
        self.max_seq_len = args.max_seq_len
        
        # Compute position embeddings
        inv_freq = 1.0 / (args.rope_theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Apply scaling for extended context
        if args.max_seq_len > args.original_seq_len:
            scale = math.log(args.rope_factor) / 2.0
            inv_freq = inv_freq * args.rope_factor ** (scale / self.dim)
            
        # Precompute rotary embeddings
        t = torch.arange(self.max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Register buffers for cos and sin
        cos = emb.cos()
        sin = emb.sin()
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch, seq_len, heads, head_dim]
            start_pos: Starting position for computing position embeddings
            
        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.shape[1]
        
        # Get position-specific rotary embeddings
        cos = self.cos[start_pos:start_pos + seq_len]
        sin = self.sin[start_pos:start_pos + seq_len]
        
        # Reshape for broadcasting
        cos = cos.view(1, seq_len, 1, -1)  # [1, seq_len, 1, dim]
        sin = sin.view(1, seq_len, 1, -1)  # [1, seq_len, 1, dim]
        
        # Split input into even and odd dimensions
        x_split = x.chunk(2, dim=-1)
        
        # Apply rotary embeddings
        rx = torch.cat([
            x_split[0] * cos - x_split[1] * sin,
            x_split[0] * sin + x_split[1] * cos,
        ], dim=-1)
        
        return rx


class Attention(nn.Module):
    """Multi-head attention with support for rotary embeddings and LoRA-style projections."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # Store dimensions
        self.hidden_size = args.dim
        self.lora_rank = args.kv_lora_rank
        self.mp_size = 8  # DeepSeek uses 8-way model parallel
        self.shard_size = self.lora_rank // self.mp_size  # 32 per shard
        
        # Query uses full dimension
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        # Key/Value use LoRA-style projections with model parallel sharding
        self.k_proj_a = nn.ModuleList([
            nn.Linear(self.hidden_size, self.shard_size, bias=True)
            for _ in range(self.mp_size)
        ])
        self.k_proj_b = nn.Linear(self.shard_size, self.hidden_size, bias=False)
        
        self.v_proj_a = nn.ModuleList([
            nn.Linear(self.hidden_size, self.shard_size, bias=True)
            for _ in range(self.mp_size)
        ])
        self.v_proj_b = nn.Linear(self.shard_size, self.hidden_size, bias=False)
        
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(args)
        self.scale = self.head_dim ** -0.5
        
        # Apply extended context scaling if needed
        if args.max_seq_len > args.original_seq_len:
            self.scale *= args.mscale

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        H = self.n_heads
        
        # Query projection
        q = self.q_proj(x)  # [B, T, C]
        q = q.view(B, T, H, -1)  # [B, T, H, head_dim]
        
        # Key projection with LoRA and model parallel
        k_shards = []
        for i in range(self.mp_size):
            k_shard = self.k_proj_a[i](x)  # [B, T, shard_size]
            k_shards.append(k_shard)
        k = torch.cat(k_shards, dim=-1)  # [B, T, lora_rank]
        k = self.k_proj_b(k)  # [B, T, hidden_size]
        k = k.view(B, T, H, -1)  # [B, T, H, head_dim]
        
        # Value projection with LoRA and model parallel
        v_shards = []
        for i in range(self.mp_size):
            v_shard = self.v_proj_a[i](x)  # [B, T, shard_size]
            v_shards.append(v_shard)
        v = torch.cat(v_shards, dim=-1)  # [B, T, lora_rank]
        v = self.v_proj_b(v)  # [B, T, hidden_size]
        v = v.view(B, T, H, -1)  # [B, T, H, head_dim]
        
        # Apply rotary embeddings
        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)
        
        # Compute attention
        attn = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        
        # Compute output
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        out = out.reshape(B, T, C)
        return self.o_proj(out)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Custom state dict loading to handle model parallel sharding."""
        # Let parent class handle non-sharded weights
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.dim, args.inter_dim, bias=False)
        self.up_proj = nn.Linear(args.dim, args.inter_dim, bias=False)
        self.down_proj = nn.Linear(args.inter_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = FeedForward(args)
        self.input_layernorm = RMSNorm(args.dim)
        self.post_attention_layernorm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), start_pos, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Transformer(nn.Module):
    """DeepSeek transformer model."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Custom state dict loading with key remapping."""
        # Create a copy of the state dict to modify
        new_state_dict = {}
        
        # Define key mappings (DeepSeek checkpoint -> our model)
        key_map = {
            'wq': 'q_proj',
            'wk': 'k_proj',
            'wv': 'v_proj',
            'wo': 'o_proj',
            'w1': 'gate_proj',
            'w2': 'up_proj',
            'w3': 'down_proj',
        }
        
        # Process each key in the state dict
        for key, value in state_dict.items():
            # Remove 'model.' prefix if present
            if key.startswith('model.'):
                key = key[6:]
                
            # Apply key mappings
            new_key = key
            for old, new in key_map.items():
                if old in key:
                    new_key = key.replace(old, new)
                    break
                    
            new_state_dict[prefix + new_key] = value
        
        # Load the remapped state dict
        super()._load_from_state_dict(new_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T = tokens.size()
        
        # Get embeddings
        h = self.embed_tokens(tokens)
        
        # Create attention mask
        mask = None
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, start_pos, mask)
        
        # Output projection
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits 