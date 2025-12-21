import torch
import json
import time

class DyadicMonitor:
    def __init__(self, log_path="metrics.jsonl"):
        self.log_path = log_path
        # Clear file
        with open(log_path, 'w') as f:
            pass
            
    def log_step(self, step, model, loss):
        metrics = {}
        metrics['step'] = step
        metrics['loss'] = loss
        metrics['timestamp'] = time.time()
        
        with torch.no_grad():
            # 1. Dyadic Heartbeat (Recurrence Health)
            # Check first layer
            layer0 = model.layers[0]
            # decay = num / 2^shift
            # base_decay_nums is [D_inner, D_state]
            # decay_shifts is [D_inner, D_state]
            
            # Use float calculation for monitoring
            decay_rates = layer0.base_decay_nums / (2.0 ** layer0.decay_shifts)
            metrics['dyadic/mean_decay'] = decay_rates.mean().item()
            metrics['dyadic/min_decay'] = decay_rates.min().item()
            metrics['dyadic/max_decay'] = decay_rates.max().item()
            
            # 2. Explosion Watch (Norms & Activations)
            avg_k = 0.0
            max_act = 0.0
            count = 0
            for layer in model.layers:
                avg_k += layer.norm.last_k
                if layer.last_act_max > max_act:
                    max_act = layer.last_act_max
                count += 1
            metrics['norm/avg_shift_k'] = avg_k / count
            metrics['stability/max_activation'] = max_act
            
            # 3. Ternary Health (BitNet Weights)
            # Check first layer in_proj
            w = layer0.in_proj.weight
            # Quantize to see effective zeros
            from rational_bitnet import weight_quant_ternary
            w_q, _ = weight_quant_ternary(w)
            zero_pct = (w_q == 0).float().mean().item()
            metrics['bitnet/zero_sparsity'] = zero_pct
            
            # Gradient Norms (Pulse)
            # Integers (base_decay_nums) vs Floats (norm.gamma)
            grad_int = 0.0
            if layer0.base_decay_nums.grad is not None:
                grad_int = layer0.base_decay_nums.grad.norm().item()
            metrics['grad/decay_nums'] = grad_int
            
            grad_float = 0.0
            if layer0.norm.gamma.grad is not None:
                grad_float = layer0.norm.gamma.grad.norm().item()
            metrics['grad/norm_gamma'] = grad_float
            
        # Write
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(metrics) + "\n")
            
        # Console Alert
        self.print_alert(metrics)
        
    def print_alert(self, m):
        # "Geiger Counter" Logic
        print(f"  [Monitor] Decay: {m['dyadic/mean_decay']:.4f} | Shift K: {m['norm/avg_shift_k']:.2f} | MaxAct: {m['stability/max_activation']:.1f} | Zeros: {m['bitnet/zero_sparsity']:.1%}")
        
        if m['norm/avg_shift_k'] > 10:
            print("  🚨 CRITICAL: Shift K > 10. Explosion Imminent!")
        if m['bitnet/zero_sparsity'] > 0.95:
            print("  ⚠️  WARNING: Sparsity > 95%. Model Collapse risk.")
        if m['dyadic/mean_decay'] < 0.1 or m['dyadic/mean_decay'] > 0.99:
             print("  ⚠️  WARNING: Decay rate collapse (All 0 or All 1).")
