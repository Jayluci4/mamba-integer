"""
muP (Maximal Update Parameterization) for Mamba-Integer.

Enables hyperparameter transfer from small to large models:
- Train 5M model to find optimal LR, weight decay, etc.
- Transfer those HPs directly to 50M model with ZERO tuning

Reference:
- "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
  (Yang et al., 2022 - now standard in GPT-4, Llama, Mamba training)
- Cerebras-GPT validated muP at 13B scale

Key changes from standard parameterization (SP):
1. Embedding: multiply output by 1 (not 1/sqrt(d))  -- already done
2. Hidden layers: scale LR by 1/width (wider = smaller LR per param)
3. Output logits: scale by 1/width (prevents logit explosion at scale)
4. Attention/SSM: scale by 1/width (keeps variance constant)
5. Init: scale by 1/sqrt(width) for hidden, 1/width for output

For Mamba-Integer specifically:
- d_model is the "width" parameter
- base_width is the reference model width (128 for 5M model)
- All scaling is multiplicative: width_ratio = d_model / base_width
"""

import math
import torch
import torch.nn as nn


def apply_mup_init(model, config, base_width=128):
    """Apply muP initialization to MambaIntegerModel.

    Scales parameter initialization based on model width relative
    to the base (proxy) model width.

    Args:
        model: MambaIntegerModel
        config: Model config dict
        base_width: Width of the base/proxy model (default: 128 for 5M)
    """
    d_model = config["d_model"]
    width_ratio = d_model / base_width  # >1 for larger models

    print(f"Applying muP initialization: d_model={d_model}, base_width={base_width}, ratio={width_ratio:.2f}")

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # Skip biases and 1D params

        if "embedding" in name:
            # Embedding: standard init (muP leaves embedding unchanged)
            # Already initialized by PyTorch
            pass
        elif "lm_head" in name:
            # Output projection: scale by 1/width
            # This prevents logit explosion in wider models
            nn.init.normal_(param, mean=0.0, std=1.0 / (d_model * width_ratio))
            print(f"  muP output: {name} std={1.0/(d_model * width_ratio):.6f}")
        elif "in_proj" in name or "x_proj" in name or "dt_proj" in name:
            # Hidden projections: scale by 1/sqrt(fan_in)
            fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(param, mean=0.0, std=std)
        elif "out_proj" in name:
            # Output of each block: scale by 1/(width * n_layers)
            n_layer = config.get("n_layer", 4)
            std = 1.0 / (math.sqrt(d_model) * math.sqrt(n_layer))
            nn.init.normal_(param, mean=0.0, std=std)
            print(f"  muP block output: {name} std={std:.6f}")

    print(f"muP initialization complete")


def get_mup_lr_scales(model, config, base_width=128):
    """Get per-parameter learning rate scales for muP.

    In muP, wider layers need proportionally smaller learning rates
    to maintain the same effective update magnitude.

    Returns parameter groups compatible with optimizer creation.

    Args:
        model: MambaIntegerModel
        config: Model config dict
        base_width: Width of the base/proxy model

    Returns:
        List of parameter group dicts with scaled learning rates
    """
    d_model = config["d_model"]
    width_ratio = d_model / base_width

    train_cfg = config.get("training", {})
    base_lr = train_cfg.get("learning_rate", 1e-3)
    decay_lr = train_cfg.get("decay_lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.01)

    # muP parameter groups
    embedding_params = []  # LR = base_lr (no scaling)
    hidden_params = []     # LR = base_lr / width_ratio
    output_params = []     # LR = base_lr / width_ratio
    decay_params = []      # LR = decay_lr (SSM-specific, no muP scaling)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "decay_logit" in name:
            decay_params.append(param)
        elif "embedding" in name:
            embedding_params.append(param)
        elif "lm_head" in name:
            output_params.append(param)
        else:
            hidden_params.append(param)

    groups = [
        {
            "params": decay_params,
            "lr": decay_lr,
            "weight_decay": 0.0,
            "name": "decay",
        },
        {
            "params": embedding_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "embedding",
        },
        {
            "params": hidden_params,
            "lr": base_lr / width_ratio,
            "weight_decay": weight_decay,
            "name": "hidden",
        },
        {
            "params": output_params,
            "lr": base_lr / width_ratio,
            "weight_decay": weight_decay,
            "name": "output",
        },
    ]

    # Filter out empty groups
    groups = [g for g in groups if g["params"]]

    print(f"muP LR scaling (width_ratio={width_ratio:.2f}):")
    for g in groups:
        print(f"  {g['name']}: lr={g['lr']:.2e}, params={len(g['params'])}")

    return groups
