import torch
import math
import warnings
from typing import Tuple

# Only import triton if CUDA is available
USE_TRITON = False
if torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl
        USE_TRITON = True
    except:
        warnings.warn("Triton import failed, falling back to CPU implementation")

# CPU implementations
def act_quant_cpu(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU implementation of activation quantization.
    
    Args:
        x: Input tensor to quantize
        block_size: Size of blocks for quantization
        
    Returns:
        Tuple of (quantized tensor, scale factors)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, f"Last dimension must be divisible by block_size ({block_size})"
    
    # Reshape for block-wise processing
    x_reshaped = x.view(-1, block_size)
    
    # Compute scaling factors
    s = torch.max(torch.abs(x_reshaped), dim=1)[0] / 448.
    
    # Scale and quantize
    y = (x_reshaped / s.unsqueeze(1))
    if hasattr(torch, 'float8_e4m3fn'):
        y = y.to(torch.float8_e4m3fn)
    else:
        y = y.to(torch.float16)  # Fallback to float16
        
    return y.view_as(x), s.view(*x.size()[:-1], -1)

def weight_dequant_cpu(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    CPU implementation of weight dequantization.
    
    Args:
        x: Quantized weight tensor
        s: Scale factors
        block_size: Size of blocks for dequantization
        
    Returns:
        Dequantized tensor
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    M, N = x.size()
    n_blocks = (N + block_size - 1) // block_size
    
    # Expand scales to match weight dimensions
    s_expanded = s.unsqueeze(-1).expand(-1, -1, block_size)
    s_expanded = s_expanded[:, :n_blocks].reshape(M, -1)[:, :N]
    
    # Dequantize
    return x.float() * s_expanded

def fp8_gemm_cpu(a: torch.Tensor, a_scale: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """
    CPU implementation of FP8 matrix multiplication.
    
    Args:
        a: First input tensor
        a_scale: Scale factors for first tensor
        b: Second input tensor
        b_scale: Scale factors for second tensor
        
    Returns:
        Result of matrix multiplication
    """
    # Dequantize inputs
    a_dequant = weight_dequant_cpu(a, a_scale) if hasattr(a, 'scale') else a
    b_dequant = weight_dequant_cpu(b, b_scale) if hasattr(b, 'scale') else b
    
    # Perform matrix multiplication
    return torch.matmul(a_dequant, b_dequant)

# Main interface functions that choose between CPU and GPU implementations
def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes activations using block-wise scaling.
    
    Args:
        x: Input tensor to quantize
        block_size: Size of blocks for quantization
        
    Returns:
        Tuple of (quantized tensor, scale factors)
    """
    if USE_TRITON and x.is_cuda:
        return act_quant_triton(x, block_size)
    return act_quant_cpu(x, block_size)

def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes weights using scaling factors.
    
    Args:
        x: Quantized weight tensor
        s: Scale factors
        block_size: Size of blocks for dequantization
        
    Returns:
        Dequantized tensor
    """
    if USE_TRITON and x.is_cuda:
        return weight_dequant_triton(x, s, block_size)
    return weight_dequant_cpu(x, s, block_size)

def fp8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with FP8 support.
    
    Args:
        a: First input tensor
        b: Second input tensor
        
    Returns:
        Result of matrix multiplication
    """
    if USE_TRITON and a.is_cuda and b.is_cuda:
        return fp8_gemm_triton(a, b)
    return fp8_gemm_cpu(a, getattr(a, 'scale', None), b, getattr(b, 'scale', None))

# Triton implementations
if USE_TRITON:
    @triton.jit
    def act_quant_kernel(
        x_ptr, y_ptr, s_ptr,
        n_elements, block_size,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton kernel for activation quantization."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        
        # Load input block
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Compute scale
        x_abs = tl.abs(x)
        s = tl.max(x_abs) / 448.
        
        # Quantize
        y = x / s
        y = tl.where(mask, y, 0.0)
        
        # Store results
        tl.store(y_ptr + offsets, y, mask=mask)
        if pid % (n_elements // block_size) == 0:
            tl.store(s_ptr + pid // (n_elements // block_size), s)

    @triton.jit
    def weight_dequant_kernel(
        x_ptr, s_ptr, y_ptr,
        M, N, block_size,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr
    ):
        """Triton kernel for weight dequantization."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Load input blocks
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        
        x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
        s = tl.load(s_ptr + offs_m * (N // block_size) + offs_n // block_size)
        
        # Dequantize
        y = x * s
        
        # Store results
        tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def fp8_gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """Triton kernel for FP8 matrix multiplication."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        
        # Compute block indices
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        
        # Compute offsets
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # Compute pointers
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        # Initialize accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Iterate over k dimension
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load blocks
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            
            # Update accumulator
            accumulator += tl.dot(a, b)
            
            # Update pointers
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        # Store results
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    def act_quant_triton(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU implementation of activation quantization using Triton."""
        n_elements = x.numel()
        y = torch.empty_like(x)
        s = torch.empty((n_elements // block_size,), dtype=torch.float32, device=x.device)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        act_quant_kernel[grid](x, y, s, n_elements, block_size, BLOCK_SIZE=1024)
        
        return y, s

    def weight_dequant_triton(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        """GPU implementation of weight dequantization using Triton."""
        M, N = x.size()
        y = torch.empty_like(x, dtype=torch.float32)
        
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N'])
        )
        weight_dequant_kernel[grid](x, s, y, M, N, block_size, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128)
        
        return y

    def fp8_gemm_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """GPU implementation of FP8 matrix multiplication using Triton."""
        M, K = a.size()
        _, N = b.size()
        c = torch.empty((M, N), dtype=torch.float32, device=a.device)
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
        fp8_gemm_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1)
        )
        
        return c 