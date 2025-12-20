
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import ctypes
import os
from typing import Tuple, List

# Try loading CUDA library via ctypes
HAS_CUDA_EXT = False
libdyadic = None

try:
    # Path relative to this file
    base_path = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(base_path, "cuda", "libdyadic.so")
    
    if os.path.exists(lib_path):
        libdyadic = ctypes.CDLL(lib_path)
        HAS_CUDA_EXT = True
        print(f"Dyadic CUDA library loaded from {lib_path}")
        
        libdyadic.launch_dyadic_transform.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool
        ]
        
        class DyadicCUDAFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x_re, x_im, lambdas, gammas, negates, scale_bits, inverse):
                ctx.lambdas = lambdas
                ctx.gammas = gammas
                ctx.negates = negates
                ctx.scale_bits = scale_bits
                ctx.inverse = inverse
                ctx.n = x_re.shape[-1]
                
                out_re = torch.empty_like(x_re)
                out_im = torch.empty_like(x_im)
                
                # Flatten
                batch_size = x_re.numel() // ctx.n
                x_re_contig = x_re.contiguous()
                x_im_contig = x_im.contiguous()
                
                libdyadic.launch_dyadic_transform(
                    ctypes.c_void_p(x_re_contig.data_ptr()),
                    ctypes.c_void_p(x_im_contig.data_ptr()),
                    ctypes.c_void_p(out_re.data_ptr()),
                    ctypes.c_void_p(out_im.data_ptr()),
                    ctypes.c_void_p(lambdas.data_ptr()),
                    ctypes.c_void_p(gammas.data_ptr()),
                    ctypes.c_void_p(negates.data_ptr()),
                    ctypes.c_int(batch_size),
                    ctypes.c_int(ctx.n),
                    ctypes.c_int(scale_bits),
                    ctypes.c_bool(inverse)
                )
                return out_re, out_im

            @staticmethod
            def backward(ctx, grad_out_re, grad_out_im):
                # Gradient of Forward Transform is Inverse Transform
                # Gradient of Inverse Transform is Forward Transform
                # (Because Dyadic Transform is orthogonal/unitary)
                
                # Note: We need to handle the scaling factor N if the transform is not normalized
                # Our kernel:
                # Forward: sum
                # Inverse: sum / N
                
                # If we computed y = Forward(x), then dy/dx = Forward^T
                # Forward matrix F is symmetric? 
                # DFT matrix is symmetric. 
                # F_{kj} = exp(-i 2pi k j / N)
                # F^H = exp(i 2pi k j / N) = Inverse * N
                
                # If ctx.inverse was False (Forward):
                # We need to apply Inverse Transform to gradients, but WITHOUT 1/N scaling?
                # Actually, standard DFT: y = Fx. grad_x = F^H grad_y.
                # F^H is the unnormalized inverse transform.
                
                # Our kernel's "inverse=True" does divide by N.
                # So to get unnormalized inverse, we can use inverse=True and multiply by N?
                # Or use inverse=False with "negative angles"?
                # Wait, our kernel params (lambdas/gammas) are pre-computed for a specific direction.
                
                # Let's trust the orthogonality:
                # If orthogonal, grad_x = Inverse(grad_y).
                # But our Forward is not orthonormal (it scales by sqrt(N) effectively, energy increases by N).
                # Forward: y = sum(...) -> Energy * N
                # Inverse: x = sum(...) / N
                
                # So if y = F(x), grad_x = F^T grad_y.
                # Since F is symmetric (mostly), F^T = F. 
                # Wait, F is complex symmetric? No, Hermitian for DFT?
                # DFT: F_{kj} = w^{kj}. Symmetric.
                # So F^T = F.
                # Thus grad_x = Forward(grad_y) using the SAME parameters?
                
                # Let's verify:
                # y_k = sum_n x_n W^{kn}
                # d y_k / d x_n = W^{kn}
                # dL / d x_n = sum_k (dL / d y_k) * (d y_k / d x_n) = sum_k grad_y_k * W^{kn}
                # This is exactly Forward Transform of grad_y!
                
                # So for DFT, backward of Forward is Forward!
                # backward of Inverse is Inverse?
                
                # Let's re-check.
                # If ctx.inverse is False (Forward transform params):
                # We want to apply the same transform?
                # Params are for exp(-i ...).
                # We need sum_k grad_y_k * exp(-i 2pi k n / N).
                # Yes, it's the same transform.
                
                # However, for Inverse (ctx.inverse=True):
                # y = sum x / N.
                # dy/dx = 1/N * W^{-kn}.
                # dL/dx = sum grad_y * 1/N * W^{-kn}.
                # This is Inverse transform of grad_y.
                
                # So simpler than expected:
                # Backward pass uses the SAME flags/params as Forward pass.
                
                grad_in_re = torch.empty_like(grad_out_re)
                grad_in_im = torch.empty_like(grad_out_im)
                
                batch_size = grad_out_re.numel() // ctx.n
                
                # Use SAME inverse flag
                libdyadic.launch_dyadic_transform(
                    ctypes.c_void_p(grad_out_re.contiguous().data_ptr()),
                    ctypes.c_void_p(grad_out_im.contiguous().data_ptr()),
                    ctypes.c_void_p(grad_in_re.data_ptr()),
                    ctypes.c_void_p(grad_in_im.data_ptr()),
                    ctypes.c_void_p(ctx.lambdas.data_ptr()),
                    ctypes.c_void_p(ctx.gammas.data_ptr()),
                    ctypes.c_void_p(ctx.negates.data_ptr()),
                    ctypes.c_int(batch_size),
                    ctypes.c_int(ctx.n),
                    ctypes.c_int(ctx.scale_bits),
                    ctypes.c_bool(ctx.inverse) 
                )
                
                return grad_in_re, grad_in_im, None, None, None, None, None

    else:
        print(f"CUDA library not found at {lib_path}. Using PyTorch fallback.")
except Exception as e:
    print(f"Failed to load CUDA library: {e}. Using PyTorch fallback.")

class DyadicComplexRotation(nn.Module):
    """
    Rotates a complex vector (x + iy) by angle theta using 
    the 3-shear lifting scheme with dyadic rational coefficients.
    Includes angle reduction to avoidance singularities at pi.
    """
    def __init__(self, angle: float, bits: int = 16):
        super().__init__()
        
        # Angle Reduction to [-pi/2, pi/2]
        self.negate = False
        
        # Normalize to [-pi, pi]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        
        if angle > math.pi / 2:
            angle -= math.pi
            self.negate = True
        elif angle < -math.pi / 2:
            angle += math.pi
            self.negate = True
            
        self.angle = angle
        
        # Compute lifting coefficients
        t = math.tan(angle / 2)
        s = math.sin(angle)
        
        # Quantize to Dyadic Rationals (n / 2^k)
        self.scale = 2**20
        self.lambda_val = round(-t * self.scale)
        self.gamma_val = round(s * self.scale)
        
    def forward(self, x, y):
        # 1. Shear X
        x = x + (y * self.lambda_val) / self.scale
        # 2. Shear Y
        y = y + (x * self.gamma_val) / self.scale
        # 3. Shear X
        x = x + (y * self.lambda_val) / self.scale
        
        if self.negate:
            return -x, -y
        else:
            return x, y

class DyadicTransform1D(nn.Module):
    """Computes Dyadic-Cayley Transform (DFT-like) along last dimension."""
    def __init__(self, n: int, inverse: bool = False):
        super().__init__()
        self.n = n
        self.inverse = inverse
        
        # Always initialize PyTorch fallback rotations for CPU execution
        self.rotations = nn.ModuleList()
        for k in range(n):
            sign = 1 if inverse else -1
            freq_rotations = nn.ModuleList()
            for i in range(n):
                angle = sign * 2 * math.pi * k * i / n
                freq_rotations.append(DyadicComplexRotation(angle))
            self.rotations.append(freq_rotations)

        if HAS_CUDA_EXT:
            # Pre-compute parameters tensors for CUDA
            # We need tensors of shape [N, N] for lambdas, gammas, negates
            # The kernel will access param[k, i]
            self.scale_bits = 20
            self.lambdas = torch.zeros(n, n, dtype=torch.int32)
            self.gammas = torch.zeros(n, n, dtype=torch.int32)
            self.negates = torch.zeros(n, n, dtype=torch.bool)
            
            for k in range(n):
                sign = 1 if inverse else -1
                for i in range(n):
                    angle = sign * 2 * math.pi * k * i / n
                    
                    # Manual logic duplication for pre-computation
                    negate = False
                    angle_norm = (angle + math.pi) % (2 * math.pi) - math.pi
                    if angle_norm > math.pi / 2:
                        angle_norm -= math.pi
                        negate = True
                    elif angle_norm < -math.pi / 2:
                        angle_norm += math.pi
                        negate = True
                    
                    t = math.tan(angle_norm / 2)
                    s = math.sin(angle_norm)
                    scale = 2**self.scale_bits
                    
                    self.lambdas[k, i] = round(-t * scale)
                    self.gammas[k, i] = round(s * scale)
                    self.negates[k, i] = negate
            
            # Move to CUDA lazily in forward
            self.params_cuda = False
            
        else:
            # Pre-compute rotations for each frequency k (PyTorch ModuleList)
            self.rotations = nn.ModuleList()
            for k in range(n):
                # DFT: exp(-i * 2pi * k * n / N)
                # Inverse: exp(+i * 2pi * k * n / N)
                sign = 1 if inverse else -1
                
                freq_rotations = nn.ModuleList()
                for i in range(n):
                    angle = sign * 2 * math.pi * k * i / n
                    freq_rotations.append(DyadicComplexRotation(angle))
                self.rotations.append(freq_rotations)
            
    def forward(self, x_re, x_im=None):
        # x_re: [..., N]
        if x_im is None:
            x_im = torch.zeros_like(x_re)
            
        if HAS_CUDA_EXT and x_re.is_cuda:
            # Ensure params are on correct device
            if not self.params_cuda:
                device = x_re.device
                self.lambdas = self.lambdas.to(device)
                self.gammas = self.gammas.to(device)
                self.negates = self.negates.to(device)
                self.params_cuda = True
                
            out_re = torch.empty_like(x_re)
            out_im = torch.empty_like(x_im)
            
            # Flatten batch dims
            batch_size = x_re.numel() // self.n
            
            # Ensure contiguous memory for pointers
            x_re_contig = x_re.contiguous()
            x_im_contig = x_im.contiguous()
            
            # Use Autograd Function
            return DyadicCUDAFunction.apply(
                x_re_contig, x_im_contig,
                self.lambdas, self.gammas, self.negates,
                self.scale_bits, self.inverse
            )
            
        else:
            # Fallback (Slow Python Loop)
            out_re_list = []
            out_im_list = []
            
            # Iterate over output frequencies k
            for k in range(self.n):
                acc_re = torch.zeros_like(x_re[..., 0])
                acc_im = torch.zeros_like(x_re[..., 0])
                
                # Sum over input time/space i
                for i in range(self.n):
                    rot = self.rotations[k][i]
                    r, m = rot(x_re[..., i], x_im[..., i])
                    acc_re += r
                    acc_im += m
                
                if self.inverse:
                    acc_re = acc_re / self.n
                    acc_im = acc_im / self.n
                    
                out_re_list.append(acc_re)
                out_im_list.append(acc_im)
                
            return torch.stack(out_re_list, dim=-1), torch.stack(out_im_list, dim=-1)

class DyadicWinograd2D(nn.Module):
    """
    2D Dyadic Winograd Convolution Layer F(m, r).
    Replaces Conv2d(3x3).
    """
    def __init__(self, in_channels, out_channels, m=8, r=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m = m
        self.r = r
        self.n = m + r - 1 # Tile size (e.g., 10 for F(8,3))
        self.stride = stride
        self.padding = padding
        
        # Transforms
        self.transform = DyadicTransform1D(self.n, inverse=False)
        self.inverse_transform = DyadicTransform1D(self.n, inverse=True)
        
        # Weights (Frequency Domain)
        # Shape: [Out, In, N, N] (2D transform of 3x3 kernel)
        self.weight_re = nn.Parameter(torch.zeros(out_channels, in_channels, self.n, self.n))
        self.weight_im = nn.Parameter(torch.zeros(out_channels, in_channels, self.n, self.n))
        
    def set_spatial_weights(self, weight_tensor):
        """
        Convert standard 3x3 spatial weights to Dyadic Frequency Domain.
        weight_tensor: [Out, In, 3, 3]
        """
        with torch.no_grad():
            # Flip weights to match Conv2d (Correlation) using DFT (Convolution)
            weight_tensor = weight_tensor.flip([2, 3])
            
            # 1. Pad to N x N
            # Winograd kernel transform G is usually [N x r]
            # We place the 3x3 kernel in the top-left (or consistent location)
            # Standard practice: 0-pad to right/bottom
            w_pad = F.pad(weight_tensor, (0, self.n - 3, 0, self.n - 3)) # [Out, In, N, N]
            
            # Center kernel for cyclic convolution (roll by -1)
            # For 3x3 kernel, index 0 is left (-1), 1 is center (0), 2 is right (+1)
            # In cyclic buffer, we want -1 at index N-1, 0 at 0, 1 at 1.
            # Roll -1 moves 1 to 0, 2 to 1, 0 to 9. Perfect.
            w_pad = torch.roll(w_pad, shifts=(-1, -1), dims=(-2, -1))
            
            # 2. Transform Rows
            # Treat as N independent 1D signals of length N
            # Flatten to [Out*In*N, N]
            w_flat = w_pad.view(-1, self.n)
            w_row_re, w_row_im = self.transform(w_flat)
            
            # 3. Transform Cols
            # Transpose to [..., N, N] -> [..., N, N] (swap last two)
            w_row_re = w_row_re.view(self.out_channels, self.in_channels, self.n, self.n).permute(0, 1, 3, 2)
            w_row_im = w_row_im.view(self.out_channels, self.in_channels, self.n, self.n).permute(0, 1, 3, 2)
            
            w_flat_col_re = w_row_re.reshape(-1, self.n)
            w_flat_col_im = w_row_im.reshape(-1, self.n)
            
            w_col_re, w_col_im = self.transform(w_flat_col_re, w_flat_col_im)
            
            # Transpose back
            self.weight_re.data = w_col_re.view(self.out_channels, self.in_channels, self.n, self.n).permute(0, 1, 3, 2)
            self.weight_im.data = w_col_im.view(self.out_channels, self.in_channels, self.n, self.n).permute(0, 1, 3, 2)
            
            print(f"Dyadic weights initialized. Shape: {self.weight_re.shape}")

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 1. Padding
        # Original padding (usually 1)
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        B, C, Hp, Wp = x_padded.shape
        
        # Calculate how many tiles we need to cover Hp, Wp
        # Each tile covers n pixels, and we jump by m.
        # We need (num_tiles - 1) * m + n >= Hp
        num_tiles_h = max(1, math.ceil((Hp - self.n) / self.m) + 1)
        num_tiles_w = max(1, math.ceil((Wp - self.n) / self.m) + 1)
        
        # Required padded size for these tiles
        Hp_req = (num_tiles_h - 1) * self.m + self.n
        Wp_req = (num_tiles_w - 1) * self.m + self.n
        
        extra_h = Hp_req - Hp
        extra_w = Wp_req - Wp
        
        # Pad right and bottom to satisfy tile requirements
        if extra_h > 0 or extra_w > 0:
            x_padded = F.pad(x_padded, (0, extra_w, 0, extra_h))
            
        # 2. Tiling (Unfold)
        # Use padding=0 because we already padded x_padded
        unfold = nn.Unfold(kernel_size=self.n, stride=self.m, padding=0)
        x_unfold = unfold(x_padded) # [B, C*N*N, L] where L is num_tiles
        L = x_unfold.shape[-1]
        
        # Reshape to [B, C, N, N, L] -> [B, L, C, N, N]
        x_tiles = x_unfold.view(B, C, self.n, self.n, L).permute(0, 4, 1, 2, 3)
        
        # Reshape for Batch Transform: [B*L*C*N, N] (Transform rows)
        x_flat = x_tiles.reshape(-1, self.n)
        
        # 1. Transform Rows
        t_row_re, t_row_im = self.transform(x_flat)
        
        # 2. Transform Cols
        # Reshape [B, L, C, N, N]
        t_row_re = t_row_re.view(B, L, C, self.n, self.n).permute(0, 1, 2, 4, 3) # Swap H,W for col transform
        t_row_im = t_row_im.view(B, L, C, self.n, self.n).permute(0, 1, 2, 4, 3)
        
        t_flat_re = t_row_re.reshape(-1, self.n)
        t_flat_im = t_row_im.reshape(-1, self.n)
        
        t_col_re, t_col_im = self.transform(t_flat_re, t_flat_im)
        
        # Transpose back: [B, L, C, N, N] (Frequency Domain Tiles)
        X_re = t_col_re.view(B, L, C, self.n, self.n).permute(0, 1, 2, 4, 3)
        X_im = t_col_im.view(B, L, C, self.n, self.n).permute(0, 1, 2, 4, 3)
        
        # 3. Hadamard Product
        # Reshape X: [B*L, In, N, N]
        X_re_reshaped = X_re.reshape(-1, self.in_channels, self.n, self.n)
        X_im_reshaped = X_im.reshape(-1, self.in_channels, self.n, self.n)
        
        # Use distinct indices for N, N dimensions to avoid ambiguity
        # b=batch, i=in_channels, o=out_channels, x=freq_row, y=freq_col
        Y_re = torch.einsum('bixy,oixy->boxy', X_re_reshaped, self.weight_re) - \
               torch.einsum('bixy,oixy->boxy', X_im_reshaped, self.weight_im)
               
        Y_im = torch.einsum('bixy,oixy->boxy', X_re_reshaped, self.weight_im) + \
               torch.einsum('bixy,oixy->boxy', X_im_reshaped, self.weight_re)
               
        # 4. Inverse Transform Rows
        y_flat = Y_re.reshape(-1, self.n)
        y_flat_im = Y_im.reshape(-1, self.n)
        
        i_row_re, i_row_im = self.inverse_transform(y_flat, y_flat_im)
        
        # 5. Inverse Transform Cols
        i_row_re = i_row_re.view(B*L, self.out_channels, self.n, self.n).permute(0, 1, 3, 2)
        i_row_im = i_row_im.view(B*L, self.out_channels, self.n, self.n).permute(0, 1, 3, 2)
        
        i_flat_re = i_row_re.reshape(-1, self.n)
        i_flat_im = i_row_im.reshape(-1, self.n)
        
        i_col_re, _ = self.inverse_transform(i_flat_re, i_flat_im) 
        
        # Transpose back
        out_tiles = i_col_re.view(B, L, self.out_channels, self.n, self.n).permute(0, 1, 2, 4, 3)
        
        # 6. Extract Valid Output (m x m)
        start = (self.r - 1) // 2
        end = start + self.m
        out_valid = out_tiles[..., start:end, start:end] # [B, L, C, m, m]
        
        # 7. Fold (Stitch tiles back)
        H_fold = num_tiles_h * self.m
        W_fold = num_tiles_w * self.m
        
        # Reshape out_valid: [B, L, C, m, m] -> [B, C*m*m, L]
        out_ready = out_valid.permute(0, 2, 3, 4, 1).reshape(B, self.out_channels * self.m * self.m, L)
        
        fold = nn.Fold(output_size=(H_fold, W_fold), kernel_size=self.m, stride=self.m)
        output = fold(out_ready)
             
        # 8. Crop to original spatial size
        return output[:, :, :H, :W]


