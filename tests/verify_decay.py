
import torch
import math

def verify_decay_precision():
    print("--- Verifying Dyadic Decay Precision ---")
    print("Goal: Can (num / 2^shift) approximate exp(-delta) accurately?")
    
    scale_bits = 15
    scale = 2**scale_bits
    
    # Test typical Mamba decay ranges (Delta * A) -> exp(-dt * A)
    # usually decays are in range [0.001, 1.0]
    
    targets = [0.9, 0.99, 0.999, 0.5, 0.1, 0.01]
    
    print(f"\nUsing Fixed Point Scale: 2^{scale_bits} ({scale})")
    print(f"{ 'Target':<10} | {'Approx':<10} | {'Num':<6} | {'Error':<10}")
    print("-" * 50)
    
    for target in targets:
        # Find best num
        num = round(target * scale)
        approx = num / scale
        error = abs(target - approx)
        
        print(f"{target:<10.4f} | {approx:<10.4f} | {num:<6} | {error:<10.6f}")
        
    print("\nConclusion:")
    max_err = 1.0 / scale
    print(f"Max possible error is 1/2^{scale_bits} = {max_err:.6f}")
    if max_err < 1e-4:
        print("Precision is better than 1e-4. Sufficient for Neural Networks.")
    else:
        print("Precision might be too low.")

if __name__ == "__main__":
    verify_decay_precision()
