import torch
import torch.nn as nn
import math

def get_hippo_s4d_real(d_state):
    """
    Generate the S4D-Real initialization for the A matrix (diagonal).
    A_n = -(n + 1) for n in 0..d_state-1
    This ensures a range of decay timescales from fast to slow.
    """
    # Range from 1.0 to d_state
    # Standard S4D-Real: -0.5 * (1, 2, ..., N) ?
    # Or just -(1, ..., N).
    # Mamba 1.0 uses: A_init = repeat(arange(1, d_state+1))
    # Then A_log = log(A_init).
    # So A = - (1..d_state).
    
    return -torch.arange(1, d_state + 1, dtype=torch.float64)

def project_to_dyadic(A_ref, dt_min=0.001, dt_max=0.1, scale_bits=15):
    """
    Project continuous A_ref onto the Dyadic Grid (num, shift).
    
    A_discrete = exp(A_ref * dt)
    We need to initialize 'nums' such that: num / 2^shift \approx A_discrete
    
    Problem: 'dt' is dynamic in Mamba (input dependent).
    In "Training from Scratch" with Dyadic Mamba, we typically fix 'shift' globally
    and learn 'num' (which represents the effective decay per step).
    
    However, if the architecture keeps the 'dt' branch, then 'num' depends on 'dt'.
    
    IF the architecture is "Pure Dyadic Mamba" (Horizon 1):
    Recurrence: h = (h * num) >> shift + x
    Here 'num' IS the decay factor.
    
    So we need to bake an "average dt" into the initialization to get the starting nums.
    Expected dt range is [0.001, 0.1]. exp(-0.001) ~ 0.999. exp(-0.1) ~ 0.9.
    
    We calculate A_discrete using a geometric mean dt or random sampling?
    Let's use a standard expected dt=0.01 (log-space middle).
    """
    
    # 1. Assume canonical dt for initialization
    dt = 0.01 
    
    # 2. Compute Target Discrete Decay: D = exp(A * dt)
    # A_ref is negative. D is in (0, 1).
    decay_target = torch.exp(A_ref * dt)
    
    # 3. Project to Dyadic
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
    """
    print(f"Initializing Dyadic Layer (State={d_state}, Bits={scale_bits})...")
    
    # 1. Generate Target
    A_ref = get_hippo_s4d_real(d_state)
    
    # 2. Project
    nums, shifts = project_to_dyadic(A_ref, scale_bits=scale_bits)
    
    # 3. Inject
    # dyadic_layer.decay_nums should be a parameter of shape [D_model, D_state] 
    # or just [D_state] broadcasted? 
    # In Mamba, A is [D_inner, D_state].
    # We repeat our 1D S4D pattern across D_inner.
    
    # Check shape of target parameter
    if not hasattr(dyadic_layer, 'decay_nums'):
        print("Error: Layer has no 'decay_nums' parameter.")
        return
        
    target_shape = dyadic_layer.decay_nums.shape
    # Usually [D_inner, D_state]
    
    if len(target_shape) == 2:
        # Broadcast A_ref [D_state] to [D_inner, D_state]
        # nums is [D_state]
        nums_broad = nums.unsqueeze(0).expand(target_shape[0], -1)
        shifts_broad = shifts.unsqueeze(0).expand(target_shape[0], -1)
        
        with torch.no_grad():
            dyadic_layer.decay_nums.data.copy_(nums_broad)
            dyadic_layer.decay_shifts.data.copy_(shifts_broad)
            
    print(f"  Initialized {target_shape} from HiPPO S4D-Real.")

if __name__ == "__main__":
    # Test the projector
    print("--- Testing Dyadic HiPPO Projector ---")
    d_state = 16
    A = get_hippo_s4d_real(d_state)
    nums, shifts = project_to_dyadic(A)
    
    print("HiPPO A (Continuous):", A[:5])
    print("Projected Nums (Dyadic):", nums[:5])
    print("Implicit Shifts:", shifts[:5])
    
    # Check reconstruction
    rec = nums.float() / (2.0 ** shifts.float())
    target = torch.exp(A * 0.01)
    err = (rec - target).abs().max()
    print(f"Max Projection Error: {err.item():.6f}")