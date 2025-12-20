#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Square Activation with Downscaling
// y = (x * x) >> k
// Helps control the "Explosion" risk of x^2.

__global__ void square_activation_kernel(
    const float* __restrict__ x,      // Input
    float* __restrict__ out,          // Output
    int total_elements,
    int shift_k)                      // Downshift to counter expansion
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    float val = x[idx];
    
    // Square
    float sq = val * val;
    
    // Apply shift (division by 2^k)
    // In ZK, this is free bit-reinterpretation if aligned, or a Mul.
    float scale = 1.0f / powf(2.0f, (float)shift_k);
    
    out[idx] = sq * scale;
}

extern "C" {
    void launch_square_activation(
        float* x,
        float* out,
        int batch_size,
        int seq_len,
        int dim,
        int shift_k)
    {
        int total_elements = batch_size * seq_len * dim;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        square_activation_kernel<<<blocks, threads>>>(
            x, out, total_elements, shift_k
        );
    }
}
