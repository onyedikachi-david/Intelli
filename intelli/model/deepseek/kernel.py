import torch
from typing import Tuple
import platform

# Only import triton on Linux platforms
USE_TRITON = platform.system() == "Linux"
try:
    if USE_TRITON:
        import triton
        import triton.language as tl
        from triton import Config
except ImportError:
    USE_TRITON = False


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors.
    
    Args:
        x_ptr: Pointer to quantized weights
        s_ptr: Pointer to scaling factors
        y_ptr: Pointer to output buffer
        M: Number of rows
        N: Number of columns
        BLOCK_SIZE: Block size for tiling
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.
    Falls back to PyTorch implementation on non-Linux platforms.
    
    Args:
        x: Quantized weight tensor of shape (M, N)
        s: Scale tensor
        block_size: Block size for dequantization (used only with Triton)
        
    Returns:
        Dequantized weight tensor
    """
    # PyTorch implementation
    M, N = x.size()
    n_blocks = (N + block_size - 1) // block_size
    s_expanded = s.unsqueeze(-1).expand(-1, -1, block_size)
    s_expanded = s_expanded[:, :n_blocks].reshape(M, -1)[:, :N]
    return x.float() * s_expanded


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes activations using block-wise scaling.
    
    Args:
        x_ptr: Pointer to input tensor
        y_ptr: Pointer to output tensor
        s_ptr: Pointer to scaling factors
        BLOCK_SIZE: Block size for quantization
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor using block-wise quantization.
    Falls back to PyTorch implementation on non-Linux platforms.
    
    Args:
        x: Input tensor to quantize
        block_size: Block size for quantization
        
    Returns:
        Tuple of (quantized tensor, scaling factors)
    """
    # PyTorch implementation
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    x_reshaped = x.view(-1, block_size)
    s = torch.max(torch.abs(x_reshaped), dim=1)[0] / 448.
    y = (x_reshaped / s.unsqueeze(1))
    if hasattr(torch, 'float8_e4m3fn'):
        y = y.to(torch.float8_e4m3fn)
    else:
        y = y.to(torch.float16)  # Fallback to float16 if float8 not available
    return y.view_as(x), s.view(*x.size()[:-1], -1)


# Optimized GEMM configurations
fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, 
           num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] 
    for block_n in [32, 64, 128] 
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr, a_s_ptr, b_s_ptr, M, N: tl.constexpr, K: tl.constexpr,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """
    Optimized matrix multiplication for FP8 tensors.
    
    Args:
        a_ptr: Pointer to first input matrix
        b_ptr: Pointer to second input matrix
        c_ptr: Pointer to output matrix
        a_s_ptr: Pointer to scaling factors for first matrix
        b_s_ptr: Pointer to scaling factors for second matrix
        M: First matrix rows
        N: Second matrix columns
        K: Common dimension
        BLOCK_SIZE_*: Block sizes for tiling
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
        
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using FP8 precision.
    Falls back to PyTorch implementation on non-Linux platforms.
    
    Args:
        a: First input matrix
        a_s: Scaling factors for first matrix
        b: Second input matrix
        b_s: Scaling factors for second matrix
        
    Returns:
        Result of matrix multiplication
    """
    # PyTorch implementation
    a_dequant = weight_dequant(a, a_s)
    b_dequant = weight_dequant(b, b_s)
    return torch.matmul(a_dequant, b_dequant)


if USE_TRITON:
    # Triton kernel definitions only when on Linux and triton is available
    @triton.jit
    def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
        """Triton kernel for weight dequantization"""
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        n = tl.cdiv(N, BLOCK_SIZE)
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs = offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        s = tl.load(s_ptr + pid_m * n + pid_n)
        y = x * s
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.jit
    def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
        """Triton kernel for activation quantization"""
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        s = tl.max(tl.abs(x)) / 448.
        y = x / s
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + pid, s)

    # Override functions with Triton implementations when available
    def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        assert x.is_contiguous() and s.is_contiguous()
        assert x.dim() == 2 and s.dim() == 2
        M, N = x.size()
        y = torch.empty_like(x, dtype=torch.get_default_dtype())
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
        weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
        return y

    def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.is_contiguous()
        assert x.size(-1) % block_size == 0
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
        act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
        return y, s

    def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
        assert a.is_contiguous() and b.is_contiguous()
        assert a_s.is_contiguous() and b_s.is_contiguous()
        K = a.size(-1)
        M = a.numel() // K
        N = b.size(0)
        c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
        fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
        return c 