/**
 * Warp-Based Parallel Scan for Mamba-Integer SSM
 *
 * Header file with kernel declarations.
 */

#ifndef WARP_SCAN_KERNEL_H
#define WARP_SCAN_KERNEL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Forward scan: h[t] = decay * h[t-1] + (1 - decay) * u[t]
 *
 * @param u           Input tensor [B, L, D]
 * @param decay_nums  Decay numerators [B, L, D] (float32, values in [0, 32000])
 * @param h           Output tensor [B, L, D]
 * @param B           Batch size
 * @param L           Sequence length
 * @param D           Feature dimension
 * @param stream      CUDA stream
 */
void launch_warp_scan_forward(
    float* u,
    float* decay_nums,
    float* h,
    int B, int L, int D,
    cudaStream_t stream);

/**
 * Backward scan for gradient computation.
 *
 * @param grad_h      Incoming gradient [B, L, D]
 * @param h           Forward output [B, L, D]
 * @param u           Forward input [B, L, D]
 * @param decay_nums  Decay numerators [B, L, D]
 * @param grad_u      Output gradient for u [B, L, D]
 * @param grad_nums   Output gradient for decay_nums [B, L, D]
 * @param B           Batch size
 * @param L           Sequence length
 * @param D           Feature dimension
 * @param stream      CUDA stream
 */
void launch_warp_scan_backward(
    float* grad_h,
    float* h,
    float* u,
    float* decay_nums,
    float* grad_u,
    float* grad_nums,
    int B, int L, int D,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // WARP_SCAN_KERNEL_H
