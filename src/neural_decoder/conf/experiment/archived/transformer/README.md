# Archived Transformer Experiments

These are older Transformer experiments that have been superseded by TCP Transformer configs.

## What's Here

- `crctc.yaml`, `crctc_only.yaml` - CR-CTC consistency regularization experiments (didn't work well)
- `crctc_sweep/` - Hyperparameter sweeps for CR-CTC
- `diphone_only.yaml` - Diphone-only auxiliary loss
- `fp16.yaml`, `fp16_only.yaml` - Mixed precision experiments
- `full.yaml` - Full transformer with all enhancements
- `intermediate_only.yaml` - Intermediate CTC loss only
- `quick/` - 100-epoch quick screens

## Results Summary

| Experiment | Final PER | Notes |
|------------|-----------|-------|
| diphone + intermediate | ~24.9% | Best non-TCP |
| base | ~25% | Solid baseline |
| CR-CTC (any variant) | 60-93% | Destroys performance |

## Current Best

Use TCP Transformer configs instead:

```bash
python scripts/train_model.py experiment=transformer/tcp/quick_xxxxl/gated
```

