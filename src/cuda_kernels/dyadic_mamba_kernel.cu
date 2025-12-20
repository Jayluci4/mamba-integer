#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Dyadic Mamba Recurrence: Integer-Only SSM
// h_t = (h_{t-1} * decay_num) >> decay_shift + x_t

__global__ void dyadic_scan_kernel(
    const float* __restrict__ x,          // Input [Batch, Seq, Dim] (treated as int/fixed-point)
    const int* __restrict__ decay_nums,   // [Batch, Seq, Dim]
    const int* __restrict__ decay_shifts, // [Batch, Seq, Dim]
    float* __restrict__ h_out,            // Output Hidden States [Batch, Seq, Dim]
    int batch_size,
    int seq_len,
    int dim,
    int scale_bits)                       // Global fixed-point scale (e.g. 20)
{
    // Parallelize over Batch and Dim
    // Sequential over Seq (Time)
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_parallel = batch_size * dim;
    
    if (idx >= total_parallel) return;
    
    int d = idx % dim;
    int b = idx / dim;
    
    // Scale for fixed point arithmetic
    // We treat float input 'x' as fixed point value: x_int = x * 2^scale_bits
    // We perform recurrence in long long (64-bit) to prevent overflow
    long long scale = 1LL << scale_bits;
    long long h_curr = 0; // Initial state h_{-1} = 0
    
    for (int t = 0; t < seq_len; t++) {
        // Flat index for time step t
        // Layout: [Batch, Seq, Dim] -> b * (S*D) + t * (D) + d
        int offset = b * seq_len * dim + t * dim + d;
        
        // 1. Load Input (Convert to Fixed Point)
        float x_val = x[offset];
        long long x_fixed = (long long)(x_val * scale);
        
        // 2. Load Parameters
        int num = decay_nums[offset];
        int shift = decay_shifts[offset];
        
        // 3. Recurrence: h_t = (h_{t-1} * num) >> shift + x_fixed
        // Note: num/2^shift represents the decay factor A (0 < A < 1)
        // Usually num < 2^shift.
        
        // Multiply
        h_curr = h_curr * num;
        
        // Shift (Arithmetic right shift to preserve sign)
        h_curr = h_curr >> shift;
        
        // Add Input
        h_curr += x_fixed;
        
        // 4. Store Output (Convert back to float for compatibility)
        h_out[offset] = (float)h_curr / scale;
    }
}

extern "C" {
    void launch_dyadic_scan(
        float* x,
        int* decay_nums,
        int* decay_shifts,
        float* h_out,
        int batch_size,
        int seq_len,
        int dim,
        int scale_bits)
    {
        int total_parallel = batch_size * dim;
        int threads = 256;
        int blocks = (total_parallel + threads - 1) / threads;
        
        dyadic_scan_kernel<<<blocks, threads>>>(
            x, decay_nums, decay_shifts, h_out,
            batch_size, seq_len, dim, scale_bits
        );
    }
}
