#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Depthwise Conv1d Step
// Update state buffer and compute output.
// state: [B, D, K] (K=4)
// weight: [D, K]
// x: [B, D] (new input)
// out: [B, D]

__global__ void conv1d_step_kernel(
    float* __restrict__ state,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size,
    int dim,
    int k_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * dim;
    
    if (idx >= total) return;
    
    int d = idx % dim;
    int b = idx / dim;
    
    // Shift state: state[b, d, 0..K-2] = state[b, d, 1..K-1]
    // Insert new: state[b, d, K-1] = x[b, d]
    
    float* s_ptr = state + (b * dim * k_size) + (d * k_size);
    const float* w_ptr = weight + d * k_size;
    
    float sum = 0.0f;
    
    // Unrolled shift for K=4
    // We can't shift easily in parallel if threads share data, but here each thread owns a channel state?
    // Yes, s_ptr is unique to (b,d).
    
    // Shift and Accumulate
    for (int i = 0; i < k_size - 1; i++) {
        s_ptr[i] = s_ptr[i+1];
        sum += s_ptr[i] * w_ptr[i];
    }
    
    // Insert new
    float val = x[idx];
    s_ptr[k_size - 1] = val;
    sum += val * w_ptr[k_size - 1];
    
    if (bias) {
        sum += bias[d];
    }
    
    out[idx] = sum;
}

extern "C" {
    void launch_conv1d_step(
        float* state,
        float* weight,
        float* bias,
        float* x,
        float* out,
        int batch_size,
        int dim,
        int k_size)
    {
        int total = batch_size * dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        conv1d_step_kernel<<<blocks, threads>>>(
            state, weight, bias, x, out, batch_size, dim, k_size
        );
    }
}
