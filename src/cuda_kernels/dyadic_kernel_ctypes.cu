#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper: 3-Shear Rotation (Integer/Fixed-Point Logic)
__device__ __forceinline__ void shear_rotate(
    float* x, 
    float* y, 
    int lambda_val, 
    int gamma_val, 
    int scale_bits) 
{
    float scale = (float)(1 << scale_bits);
    
    // 1. Shear X
    *x = *x + (*y * lambda_val) / scale;
    
    // 2. Shear Y
    *y = *y + (*x * gamma_val) / scale;
    
    // 3. Shear X
    *x = *x + (*y * lambda_val) / scale;
}

// CUDA Kernel for 1D Dyadic Transform
__global__ void dyadic_transform_kernel(
    const float* __restrict__ input_re,
    const float* __restrict__ input_im,
    float* __restrict__ output_re,
    float* __restrict__ output_im,
    const int* __restrict__ lambdas, // [N, N]
    const int* __restrict__ gammas,  // [N, N]
    const bool* __restrict__ negates, // [N, N]
    int batch_size,
    int n,
    int scale_bits,
    bool inverse) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * n;
    
    if (idx >= total_threads) return;
    
    int k = idx % n;       // Output frequency index
    int b = idx / n;       // Batch index
    
    // Pointers to this batch's data
    const float* in_re_ptr = input_re + b * n;
    const float* in_im_ptr = input_im + b * n;
    
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float x = in_re_ptr[i];
        float y = in_im_ptr[i];
        
        int param_idx = k * n + i;
        int lam = lambdas[param_idx];
        int gam = gammas[param_idx];
        bool neg = negates[param_idx];
        
        shear_rotate(&x, &y, lam, gam, scale_bits);
        
        if (neg) {
            x = -x;
            y = -y;
        }
        
        acc_re += x;
        acc_im += y;
    }
    
    if (inverse) {
        acc_re /= n;
        acc_im /= n;
    }
    
    output_re[idx] = acc_re;
    output_im[idx] = acc_im;
}

extern "C" {
    void launch_dyadic_transform(
        float* input_re,
        float* input_im,
        float* output_re,
        float* output_im,
        int* lambdas,
        int* gammas,
        bool* negates,
        int batch_size,
        int n,
        int scale_bits,
        bool inverse)
    {
        int threads = 256;
        int total_elements = batch_size * n;
        int blocks = (total_elements + threads - 1) / threads;
        
        dyadic_transform_kernel<<<blocks, threads>>>(
            input_re, input_im, output_re, output_im,
            lambdas, gammas, negates,
            batch_size, n, scale_bits, inverse
        );
        
        // Optional: Check for launch errors (sync is slow, so skip for production)
        // cudaDeviceSynchronize();
    }
}
