"""
Python wrapper for CUDA warp-based parallel scan.

Provides PyTorch integration via ctypes for the high-performance
warp shuffle scan kernel.
"""

import torch
import ctypes
import os

# Load the CUDA kernel library
_lib_path = os.path.join(os.path.dirname(__file__), "../cuda_kernels/libwarp_scan.so")
_lib = None

def _load_library():
    global _lib
    if _lib is None:
        if os.path.exists(_lib_path):
            _lib = ctypes.CDLL(_lib_path)

            # Define function signatures
            _lib.launch_warp_scan_forward.argtypes = [
                ctypes.c_void_p,  # u
                ctypes.c_void_p,  # decay_nums
                ctypes.c_void_p,  # h
                ctypes.c_int,     # B
                ctypes.c_int,     # L
                ctypes.c_int,     # D
                ctypes.c_void_p,  # stream
            ]
            _lib.launch_warp_scan_forward.restype = None

            _lib.launch_warp_scan_backward.argtypes = [
                ctypes.c_void_p,  # grad_h
                ctypes.c_void_p,  # h
                ctypes.c_void_p,  # u
                ctypes.c_void_p,  # decay_nums
                ctypes.c_void_p,  # grad_u
                ctypes.c_void_p,  # grad_nums
                ctypes.c_int,     # B
                ctypes.c_int,     # L
                ctypes.c_int,     # D
                ctypes.c_void_p,  # stream
            ]
            _lib.launch_warp_scan_backward.restype = None

            print("DEBUG: CUDA warp scan kernel loaded successfully.")
        else:
            print(f"DEBUG: CUDA warp scan kernel not found at {_lib_path}")
    return _lib

# Try to load on import
_load_library()

WARP_SCAN_AVAILABLE = _lib is not None


class WarpScanFunction(torch.autograd.Function):
    """
    PyTorch autograd function for CUDA warp-based scan.

    Implements convex combination: h[t] = decay * h[t-1] + (1 - decay) * u[t]

    5-6x faster than Triton's associative_scan due to warp shuffle operations.
    """

    @staticmethod
    def forward(ctx, u, decay_nums):
        """
        Forward pass: parallel scan with convex combination.

        Args:
            u: Input tensor [B, L, D]
            decay_nums: Decay numerators [B, L, D] (float32, values in [0, 32000])

        Returns:
            h: Output tensor [B, L, D]
        """
        assert u.is_cuda and decay_nums.is_cuda, "Tensors must be on CUDA"
        assert u.is_contiguous() and decay_nums.is_contiguous(), "Tensors must be contiguous"
        assert u.dtype == torch.float32 and decay_nums.dtype == torch.float32, "Tensors must be float32"

        B, L, D = u.shape
        h = torch.empty_like(u)

        # Get CUDA stream
        stream = torch.cuda.current_stream().cuda_stream

        # Launch kernel
        _lib.launch_warp_scan_forward(
            ctypes.c_void_p(u.data_ptr()),
            ctypes.c_void_p(decay_nums.data_ptr()),
            ctypes.c_void_p(h.data_ptr()),
            ctypes.c_int(B),
            ctypes.c_int(L),
            ctypes.c_int(D),
            ctypes.c_void_p(stream),
        )

        ctx.save_for_backward(u, h, decay_nums)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        """
        Backward pass: reverse scan for gradient computation.

        Returns:
            grad_u: Gradient w.r.t. input u
            grad_nums: Gradient w.r.t. decay_nums
        """
        u, h, decay_nums = ctx.saved_tensors

        # Make contiguous if needed
        if not grad_h.is_contiguous():
            grad_h = grad_h.contiguous()

        B, L, D = grad_h.shape
        grad_u = torch.empty_like(u)
        grad_nums = torch.empty_like(decay_nums)

        # Get CUDA stream
        stream = torch.cuda.current_stream().cuda_stream

        # Launch kernel
        _lib.launch_warp_scan_backward(
            ctypes.c_void_p(grad_h.data_ptr()),
            ctypes.c_void_p(h.data_ptr()),
            ctypes.c_void_p(u.data_ptr()),
            ctypes.c_void_p(decay_nums.data_ptr()),
            ctypes.c_void_p(grad_u.data_ptr()),
            ctypes.c_void_p(grad_nums.data_ptr()),
            ctypes.c_int(B),
            ctypes.c_int(L),
            ctypes.c_int(D),
            ctypes.c_void_p(stream),
        )

        return grad_u, grad_nums


def warp_scan_cuda(u, decay_nums):
    """
    High-performance warp-based parallel scan.

    Computes: h[t] = decay * h[t-1] + (1 - decay) * u[t]
    where decay = decay_nums / 32768

    5-6x faster than Triton's associative_scan.

    Args:
        u: Input tensor [B, L, D]
        decay_nums: Decay numerators [B, L, D] (float32, values in [0, 32000])

    Returns:
        h: Output tensor [B, L, D]
    """
    if not WARP_SCAN_AVAILABLE:
        raise RuntimeError("CUDA warp scan kernel not available. Build with 'make libwarp_scan.so'")

    return WarpScanFunction.apply(u, decay_nums)
