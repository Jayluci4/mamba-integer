#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ __forceinline__ void shear_rotate(
    float* x, 
    float* y, 
    int lambda_val, 
    int gamma_val, 
    int scale_bits) 
{
    float scale = (float)(1 << scale_bits);
    *x = *x + (*y * lambda_val) / scale;
    *y = *y + (*x * gamma_val) / scale;
    *x = *x + (*y * lambda_val) / scale;
}

// Branchless Dyadic RoPE Kernel
// Replaced `bool* negates` with `float* signs` (+1.0 or -1.0)
__global__ void dyadic_rope_kernel(
    float* q,          
    float* k,          
    const int* lambdas, 
    const int* gammas,  
    const float* signs, // CHANGED: Branchless sign multiplier
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int scale_bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_pairs = batch_size * num_heads * seq_len * half_dim;
    
    if (idx >= total_pairs) return;
    
    int d_pair = idx % half_dim;
    int s = (idx / half_dim) % seq_len;
    int h = (idx / (half_dim * seq_len)) % num_heads;
    int b = idx / (half_dim * seq_len * num_heads);
    
    long long offset = (long long)b * num_heads * seq_len * head_dim +
                       (long long)h * seq_len * head_dim +
                       (long long)s * head_dim +
                       d_pair;
                       
    float q_x = q[offset];
    float q_y = q[offset + half_dim];
    
    float k_x = k[offset];
    float k_y = k[offset + half_dim];
    
    int param_idx = s * half_dim + d_pair;
    int lam = lambdas[param_idx];
    int gam = gammas[param_idx];
    
    // Branchless Sign Application
    float sgn = signs[param_idx]; 
    
    // Apply Rotation to Q
    shear_rotate(&q_x, &q_y, lam, gam, scale_bits);
    // Unconditional multiplication avoids warp divergence
    q_x *= sgn;
    q_y *= sgn;
    
    // Apply Rotation to K
    shear_rotate(&k_x, &k_y, lam, gam, scale_bits);
    k_x *= sgn;
    k_y *= sgn;
    
    q[offset] = q_x;
    q[offset + half_dim] = q_y;
    
    k[offset] = k_x;
    k[offset + half_dim] = k_y;
}

extern "C" {
    void launch_dyadic_rope(
        float* q,
        float* k,
        int* lambdas,
        int* gammas,
        float* signs, // Changed signature
        int batch_size,
        int num_heads,
        int seq_len,
        int head_dim,
        int scale_bits)
    {
        int half_dim = head_dim / 2;
        int total_pairs = batch_size * num_heads * seq_len * half_dim;
        int threads = 256;
        int blocks = (total_pairs + threads - 1) / threads;
        
        dyadic_rope_kernel<<<blocks, threads>>>(
            q, k, lambdas, gammas, signs,
            batch_size, num_heads, seq_len, head_dim, scale_bits
        );
    }
}