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

    // Store last state if needed (optional)
    // h_out already stores history.
}

__global__ void dyadic_scan_backward_kernel(
    const float* __restrict__ grad_h,     // Incoming grad from next layer [Batch, Seq, Dim]
    const float* __restrict__ h_val,      // Forward pass values [Batch, Seq, Dim]
    const int* __restrict__ decay_nums,   // Forward params [Batch, Seq, Dim]
    const int* __restrict__ decay_shifts, // [Batch, Seq, Dim]
    float* __restrict__ grad_x,           // Output grad w.r.t input
    float* __restrict__ grad_decay_nums,  // Output grad w.r.t decay numerator
    int batch_size,
    int seq_len,
    int dim,
    int scale_bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_parallel = batch_size * dim;
    
    if (idx >= total_parallel) return;
    
    int d = idx % dim;
    int b = idx / dim;
    
    long long scale = 1LL << scale_bits;
    float inv_scale = 1.0f / (float)scale;
    
    // Reverse Recurrence accumulator
    // dL/dh_{t-1} = dL/dh_t * a_t
    // Since h_out stores the state sequence, we have access to h_{t-1} for parameter grad.
    
    // Gradient of state. Initialize with 0 (future is empty).
    // Note: grad_h is the gradient from the *layer above* at time t.
    // The total gradient dL/dh_t includes contribution from next time step (recurrence).
    double d_h_curr = 0.0; 
    
    for (int t = seq_len - 1; t >= 0; t--) {
        int offset = b * seq_len * dim + t * dim + d;
        
        // 1. Accumulate incoming gradient from layer above
        d_h_curr += grad_h[offset];
        
        // 2. Compute Gradients
        // grad_x_t = d_h_curr * (dh_t/dx_t) = d_h_curr * 1
        grad_x[offset] = (float)d_h_curr;
        
        // grad_a_t = d_h_curr * (dh_t/da_t) = d_h_curr * h_{t-1}
        // decay = num / 2^shift.
        // d(decay)/d(num) = 1/2^shift.
        // grad_num = d_h_curr * h_{t-1} * (1/scale)
        
        int shift = decay_shifts[offset];
        float decay_scale = 1.0f / powf(2.0f, (float)shift);
        
        float h_prev = (t > 0) ? h_val[offset - dim] : 0.0f;
        
        // Calculate grad w.r.t num (treated as float for training)
        float g_num = (float)(d_h_curr * h_prev * decay_scale);
        grad_decay_nums[offset] = g_num;
        
        // 3. Update Recurrence for next step (t-1)
        // d_h_{t-1} contribution = d_h_t * decay
        int num = decay_nums[offset];
        float decay = (float)num * decay_scale;
        
        d_h_curr = d_h_curr * decay;
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
    
    void launch_dyadic_scan_backward(
        float* grad_h,
        float* h_val,
        int* decay_nums,
        int* decay_shifts,
        float* grad_x,
        float* grad_decay_nums,
        int batch_size,
        int seq_len,
        int dim,
        int scale_bits)
    {
        int total_parallel = batch_size * dim;
        int threads = 256;
        int blocks = (total_parallel + threads - 1) / threads;
        
        dyadic_scan_backward_kernel<<<blocks, threads>>>(
            grad_h, h_val, decay_nums, decay_shifts, grad_x, grad_decay_nums,
            batch_size, seq_len, dim, scale_bits
        );
    }
}
