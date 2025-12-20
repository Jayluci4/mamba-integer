
import torch
import torch.nn as nn
import math

def get_hippo_s4d_real(d_state):
    """
    Generate the S4D-Real initialization for the A matrix (diagonal).
    A_n = -(n + 1) for n in 0..d_state-1
    """
    return -torch.arange(1, d_state + 1, dtype=torch.float64)

def project_to_dyadic(A_ref, dt=0.01, scale_bits=15):
    """
    Project continuous A_ref onto the Dyadic Grid (num, shift).
    
    A_discrete = exp(A_ref * dt)
    We need to initialize 'nums' such that: num / 2^shift \approx A_discrete
    """
    # Compute Target Discrete Decay: D = exp(A * dt)
    decay_target = torch.exp(A_ref * dt)
    
    # Project to Dyadic
    scale = 2.0 ** scale_bits
    
    # n = round(D * 2^K)
    nums = torch.round(decay_target * scale).int()
    
    # Clip to valid range [0, 2^K] (decay must be <= 1)
    max_val = int(scale)
    nums = torch.clamp(nums, 0, max_val)
    
    shifts = torch.full_like(nums, scale_bits)
    
    return nums, shifts

def initialize_dyadic_layer(dyadic_layer, d_state, scale_bits=15):
    """
    Apply HiPPO initialization to a Dyadic Mamba Layer.
    Assumes dyadic_layer has 'decay_nums' and 'decay_shifts' parameters.
    """
    if not hasattr(dyadic_layer, 'decay_nums'):
        return
        
    # Generate Target
    A_ref = get_hippo_s4d_real(d_state)
    
    # Project
    nums, shifts = project_to_dyadic(A_ref, scale_bits=scale_bits)
    
    target_shape = dyadic_layer.decay_nums.shape
    
    # Broadcast to layer shape [D_inner, D_state]
    if len(target_shape) == 2:
        nums_broad = nums.unsqueeze(0).expand(target_shape[0], -1)
        shifts_broad = shifts.unsqueeze(0).expand(target_shape[0], -1)
        
        with torch.no_grad():
            dyadic_layer.decay_nums.data.copy_(nums_broad)
            dyadic_layer.decay_shifts.data.copy_(shifts_broad)
            
    print(f"Initialized Dyadic Mamba Layer with HiPPO S4D-Real (State={d_state}, Bits={scale_bits})")
