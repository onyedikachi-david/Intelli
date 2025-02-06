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
    dim: int = 2048
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
    kv_lora_rank: int = 512
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
        
        # Compute dimensions for different attention components
        self.qk_nope_dim = args.qk_nope_head_dim * args.n_heads
        self.qk_rope_dim = args.qk_rope_head_dim * args.n_heads
        self.v_dim = args.v_head_dim * args.n_heads
        
        # Query uses full dimension with split between RoPE and non-RoPE parts
        self.q_proj = nn.Linear(args.dim, args.dim, bias=True)
        
        # Key/Value use LoRA-style projections for dimension reduction
        self.k_proj_a = nn.Linear(args.dim, args.kv_lora_rank, bias=True)
        self.k_proj_b = nn.Linear(args.kv_lora_rank, 256, bias=True)  # 256 = qk_nope_dim + qk_rope_dim per head
        self.k_norm = nn.LayerNorm(args.kv_lora_rank)
        
        self.v_proj_a = nn.Linear(args.dim, args.kv_lora_rank, bias=True)
        self.v_proj_b = nn.Linear(args.kv_lora_rank, 256, bias=True)  # 256 = v_head_dim per head
        self.v_norm = nn.LayerNorm(args.kv_lora_rank)
        
        self.o_proj = nn.Linear(args.dim, args.dim, bias=False)
        
        self.rope = RotaryEmbedding(args)
        self.scale = self.head_dim ** -0.5
        
        # Store rope dimensions
        self.rope_dim = args.qk_rope_head_dim
        
        # Apply extended context scaling if needed
        if args.max_seq_len > args.original_seq_len:
            self.scale *= args.mscale

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        H = self.n_heads
        
        # Query projection with split dimensions
        q = self.q_proj(x)  # [B, T, C]
        q = q.view(B, T, H, -1)  # [B, T, H, head_dim]
        
        # Key projection with LoRA
        k = self.k_proj_a(x)  # [B, T, kv_lora_rank]
        k = self.k_norm(k)
        k = self.k_proj_b(k)  # [B, T, 256]
        k = k.view(B, T, 1, 256).expand(B, T, H, 256)  # [B, T, H, 256]
        
        # Value projection with LoRA
        v = self.v_proj_a(x)  # [B, T, kv_lora_rank]
        v = self.v_norm(v)
        v = self.v_proj_b(v)  # [B, T, 256]
        v = v.view(B, T, 1, 256).expand(B, T, H, 256)  # [B, T, H, 256]
        
        # Apply rotary embeddings only to the RoPE part of query and key
        q_rope_dim = self.rope_dim * 2  # Multiply by 2 since we need pairs for rotation
        q_rope = q[..., :q_rope_dim]  # [B, T, H, rope_dim*2]
        q_rope = self.rope(q_rope, start_pos)  # Apply RoPE
        
        # Concatenate RoPE and non-RoPE parts
        if q.shape[-1] > q_rope_dim:
            q = torch.cat([q_rope, q[..., q_rope_dim:]], dim=-1)
        else:
            q = q_rope
        
        # Compute attention
        attn = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        
        # Compute output
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        out = out.reshape(B, T, C)
        return self.o_proj(out)


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