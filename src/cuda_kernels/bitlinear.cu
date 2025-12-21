#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Naive BitLinear Kernel (Float Input -> Int8 Quant -> Int8 Weight -> Float Output)
// Not optimized for Tensor Cores, but functional for POC.

__device__ __forceinline__ char quantize_val(float x, float scale) {
    // scale = 127 / max_abs_x
    float val = x * scale;
    float rounded = roundf(val);
    int i = (int)rounded;
    if (i > 127) i = 127;
    if (i < -127) i = -127;
    return (char)i;
}

__global__ void bitlinear_kernel(
    const float* __restrict__ x,      // [Batch, In]
    const char* __restrict__ w,       // [Out, In]
    float* __restrict__ y,            // [Batch, Out]
    float w_scale,
    int batch_size,
    int in_features,
    int out_features) 
{
    // Simple 1 thread per output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    
    if (idx >= total) return;
    
    int row = idx / out_features; // batch index
    int col = idx % out_features; // output feature index
    
    // 1. Calculate Dynamic Scale for this row (x)
    // In efficient impl, this is precomputed. Here we do it per thread (wasteful but simple).
    // Or we assume x is pre-quantized?
    // Let's implement fused quantization.
    
    const float* x_row = x + row * in_features;
    
    float max_abs = 0.0f;
    for (int i = 0; i < in_features; i++) {
        float abs_x = fabsf(x_row[i]);
        if (abs_x > max_abs) max_abs = abs_x;
    }
    float x_scale = 127.0f / (max_abs + 1e-9f);
    
    // 2. Dot Product
    float sum = 0.0f;
    const char* w_row = w + col * in_features;
    
    for (int i = 0; i < in_features; i++) {
        char x_q = quantize_val(x_row[i], x_scale);
        char w_q = w_row[i];
        
        // BitNet logic: add/sub
        // But w_q can be 0.
        // sum += x_q * w_q
        sum += (float)(x_q * w_q);
    }
    
    // 3. Dequantize
    // y = sum * (w_scale * (1/x_scale))
    // x_scale = 127/max. 1/x_scale = max/127.
    // scale_factor = w_scale * (max_abs / 127.0)
    
    float out_scale = w_scale * (max_abs / 127.0f);
    y[idx] = sum * out_scale;
}

extern "C" {
    void launch_bitlinear(
        float* x,
        char* w,
        float* y,
        float w_scale,
        int batch_size,
        int in_features,
        int out_features)
    {
        int total = batch_size * out_features;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        bitlinear_kernel<<<blocks, threads>>>(
            x, w, y, w_scale, batch_size, in_features, out_features
        );
    }
}
