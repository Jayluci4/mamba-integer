#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// BitShift Norm + Learnable Scalar
// y = (x >> k) * gamma
// k = floor(log2(rms))

__global__ void bitshift_norm_fwd_kernel(
    const float* __restrict__ x,      // Input [Batch, Seq, Dim] (treated as int/fixed)
    const float* __restrict__ gamma,  // Learnable Scalar [Dim]
    float* __restrict__ out,          // Output
    int batch_size,
    int seq_len,
    int dim,
    int scale_bits)
{
    // One block per token (Batch * Seq)
    int token_idx = blockIdx.x;
    if (token_idx >= batch_size * seq_len) return;
    
    // Pointer to this token's vector
    const float* x_ptr = x + token_idx * dim;
    float* out_ptr = out + token_idx * dim;
    
    // 1. Compute Sum of Squares (Variance)
    // Reduce within block
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x_ptr[i];
        sum_sq += val * val;
    }
    
    // Block Reduce
    // Assume blockDim.x is power of 2 (e.g. 256)
    // Shared memory reduction
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = sum_sq;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Leader computes shift k
    __shared__ int k_shift;
    if (tid == 0) {
        float var = sdata[0] / (float)dim;
        // Ideally we treat input as integer fixed point.
        // If x is float, x^2 is float.
        // We approximate k = log2(sqrt(var)) = 0.5 * log2(var)
        
        // ZK/Integer Logic:
        // var_int = var * (scale^2)? No, relative shift.
        // Just use float log2 for the "oracle" (during training).
        // In ZK, we bit-decompose var.
        
        if (var < 1e-9f) {
            k_shift = 0;
        } else {
            // k s.t. 2^k ~ sqrt(var)
            // k = floor(log2(sqrt(var)))
            k_shift = (int)floorf(log2f(sqrtf(var)));
        }
        // Clamp shift?
        if (k_shift < 0) k_shift = 0; // Don't shift left (amplify) unless designed
        // Standard norm divides by rms. So we want divide by 2^k.
        // x >> k.
    }
    __syncthreads();
    
    // 2. Apply Shift and Gamma
    int k = k_shift;
    float scale_factor = 1.0f / powf(2.0f, (float)k);
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x_ptr[i];
        // Simulate integer shift
        // val_shifted = floor(val / 2^k)
        // For gradient flow, we keep it float? 
        // "Integer-Only" means we should truncate.
        // But for training, maybe STE rounding?
        // Let's implement exact float division to represent shift
        float shifted = val * scale_factor;
        
        // Multiply by Learnable Scalar
        float g = gamma[i];
        out_ptr[i] = shifted * g;
    }
}

extern "C" {
    void launch_bitshift_norm(
        float* x,
        float* gamma,
        float* out,
        int batch_size,
        int seq_len,
        int dim,
        int scale_bits)
    {
        int total_tokens = batch_size * seq_len;
        int threads = 256;
        int blocks = total_tokens;
        // Shared mem size? Static 256 floats.
        
        bitshift_norm_fwd_kernel<<<blocks, threads>>>(
            x, gamma, out, batch_size, seq_len, dim, scale_bits
        );
    }
}
