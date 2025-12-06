# Stochastic Augmentation Experiments

## Overview

These experiments test **stochastic augmentation modes** where augmentation types
are randomly selected per sample, rather than applying all augmentations to all samples.

## Two Stochastic Modes

### 1. Exclusive Mode (`stochastic: exclusive`)
- **Exactly ONE** augmentation per sample
- Equal probability: 1/3 mixup, 1/3 adaptive, 1/3 white noise
- No combos, no "none" - every sample gets augmented

### 2. Independent Mode (`stochastic: independent`) 
- Each augmentation applied **independently** with P=1/3
- **Allows combos + none**
- Expected distribution:
  - ~30% get no augmentation
  - ~44% get exactly one
  - ~22% get two
  - ~4% get all three

## Noise Levels

Each augmentation is calibrated to contribute roughly the same noise magnitude when selected:

| Level | Mixup Ratio | Adaptive (base+scale) | White SD |
|-------|-------------|----------------------|----------|
| 1.0 SD | 0.7 | 0.8 + 0.4 | 1.0 |
| 1.2 SD | 0.85 | 1.0 + 0.4 | 1.2 |

## Experiment Grid (8 experiments)

| Model | Mode | Level | Experiment |
|-------|------|-------|------------|
| h1536 | exclusive | 1.0 SD | `h1536_exclusive_sd10` |
| h1536 | exclusive | 1.2 SD | `h1536_exclusive_sd12` |
| h1536 | independent | 1.0 SD | `h1536_independent_sd10` |
| h1536 | independent | 1.2 SD | `h1536_independent_sd12` |
| h2048 | exclusive | 1.0 SD | `h2048_exclusive_sd10` |
| h2048 | exclusive | 1.2 SD | `h2048_exclusive_sd12` |
| h2048 | independent | 1.0 SD | `h2048_independent_sd10` |
| h2048 | independent | 1.2 SD | `h2048_independent_sd12` |

## Running

```bash
# Single experiment
python scripts/train_model.py experiment=stochastic_aug/h1536_exclusive_sd10

# All experiments
for model in h1536 h2048; do
    for mode in exclusive independent; do
        for level in sd10 sd12; do
            python scripts/train_model.py experiment=stochastic_aug/${model}_${mode}_${level}
        done
    done
done
```

## Colab Cell

```python
experiments = [
    "h1536_exclusive_sd10", "h1536_exclusive_sd12",
    "h1536_independent_sd10", "h1536_independent_sd12",
    "h2048_exclusive_sd10", "h2048_exclusive_sd12",
    "h2048_independent_sd10", "h2048_independent_sd12",
]

for exp in experiments:
    print("=" * 60)
    print(f"Running: stochastic_aug/{exp}")
    print("=" * 60)
    !python -W ignore scripts/train_model.py experiment=stochastic_aug/{exp}
    print("\n")
```
