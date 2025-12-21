#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Rational Squareplus Activation
// y = (x + sqrt(x^2 + 4)) * 0.5
// Implemented using Newton-Raphson for RSQRT to avoid division/transcendental sqrt.
// sqrt(z) = z * rsqrt(z)
// rsqrt(z) approx via y_{n+1} = 0.5 * y_n * (3 - z * y_n^2)

__device__ __forceinline__ float division_free_rsqrt(float z) {
    // Initial guess: float magic or just a safe constant?
    // z = x^2 + 4. Min value is 4.
    // rsqrt(4) = 0.5.
    // For large z, rsqrt is small.
    // A standard fast inverse square root constant (0x5f3759df) works on float bits.
    
    // Fast Inverse Square Root (Quake III style) for initial guess
    float xhalf = 0.5f * z;
    int i = *(int*)&z;
    i = 0x5f3759df - (i >> 1);
    float y = *(float*)&i;
    
    // 2 Iterations of Newton-Raphson
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    
    return y;
}

__global__ void squareplus_kernel(
    const float* __restrict__ x,      // Input
    float* __restrict__ out,          // Output
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    float val = x[idx];
    
    // z = x^2 + 4
    float z = val * val + 4.0f;
    
    // sqrt(z) approx
    float r = division_free_rsqrt(z);
    float sqrt_z = z * r;
    
    // y = 0.5 * (x + sqrt_z)
    out[idx] = 0.5f * (val + sqrt_z);
}

extern "C" {
    void launch_squareplus(
        float* x,
        float* out,
        int batch_size,
        int seq_len,
        int dim)
    {
        int total_elements = batch_size * seq_len * dim;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        squareplus_kernel<<<blocks, threads>>>(
            x, out, total_elements
        );
    }
}
