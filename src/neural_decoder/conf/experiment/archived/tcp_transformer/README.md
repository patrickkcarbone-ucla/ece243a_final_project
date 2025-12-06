# Archived TCP Experiments

These are older TCP Transformer experiments that have been superseded by the XXXL and XXXXL configs.

## What's Here

- `base.yaml`, `curriculum.yaml`, `gated.yaml` - Original TCP experiments with base model
- `ablations/` - Ablation studies (no_gating, no_highway, etc.)
- `quick/` - 100-epoch quick screens with base model
- `quick_larger/` - Experiments with larger (480 hidden_dim) model
- `quick_xl/` - Experiments with XL (576 hidden_dim) model  
- `quick_xxl/` - Experiments with XXL (720 hidden_dim) model

## Current Best

Use the configs in `../quick_xxxl/` or `../quick_xxxxl/` instead:

```bash
# Best: XXXXL gated
python scripts/train_model.py experiment=transformer/tcp/quick_xxxxl/gated

# Alternative: XXXL gated
python scripts/train_model.py experiment=transformer/tcp/quick_xxxl/gated
```

