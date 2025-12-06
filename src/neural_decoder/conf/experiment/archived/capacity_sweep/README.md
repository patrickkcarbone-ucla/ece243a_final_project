# Model Capacity Experiments

Baseline experiments testing different GRU model capacities.

## Models (4)

| Config | Hidden | Layers | Notes |
|--------|--------|--------|-------|
| `h1024_base` | 1024 | 5 | Baseline (dual_region_6v_only) |
| `h1536_base` | 1536 | 5 | 1.5x hidden capacity |
| `h2048_base` | 2048 | 5 | 2x hidden capacity |
| `l6_base` | 1024 | 6 | +1 layer depth |

All use: 10K batches, LR 0.02→0.01, dual-region 6v-only, baseline augmentation (0.8 SD white noise)

## Run Commands

```bash
# Run single experiment
python scripts/train_model.py experiment=capacity_sweep/h1536_base

# Run all
for model in h1024 h1536 h2048 l6; do
    python scripts/train_model.py experiment=capacity_sweep/${model}_base
done
```

## Archived Experiments

Additional augmentation experiments (mixup, adaptive, combo, white noise controls) 
are archived in `archived/capacity_sweep/` for reference.
