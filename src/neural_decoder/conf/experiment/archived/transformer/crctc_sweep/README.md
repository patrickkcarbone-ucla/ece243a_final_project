# CR-CTC Sweep Experiments

Systematic exploration of CR-CTC with lighter regularization and increased model capacity.

## Naming Convention

`{model}_b{beta}_m{masks}_s{start}` where:
- `model`: base (~9.4M), med (~18M), lg (~24M)
- `b{XX}`: β_max × 100 (e.g., b05 = β=0.05)
- `m{XX}`: number of heavy masks
- `s{XX}`: CR-CTC start epoch

Controls: `{model}_ctrl` (no CR-CTC)

## Background

Initial CR-CTC experiments failed badly (60-93% PER vs 25% baseline). Hypotheses:
1. **Heavy view too aggressive**: 50 masks @ 37.5% fraction destroys signal
2. **β too high**: 0.2 may overwhelm the CTC loss for small models
3. **Started too early**: Model needs to learn basics before consistency helps

## Experiments

| Config | Model | β | Masks | Frac | Start | Ramp |
|--------|-------|---|-------|------|-------|------|
| `base_ctrl` | base | - | - | - | - | - |
| `base_b05_m10_s50` | base | 0.05 | 10 | 10% | 50 | 40 |
| `base_b10_m20_s50` | base | 0.10 | 20 | 15% | 50 | 40 |
| `base_b15_m30_s40` | base | 0.15 | 30 | 20% | 40 | 30 |
| `med_ctrl` | medium | - | - | - | - | - |
| `med_b05_m10_s50` | medium | 0.05 | 10 | 10% | 50 | 40 |
| `med_b10_m20_s50` | medium | 0.10 | 20 | 15% | 50 | 40 |
| `lg_ctrl` | large | - | - | - | - | - |
| `lg_b05_m10_s50` | large | 0.05 | 10 | 10% | 50 | 40 |
| `lg_b10_m20_s50` | large | 0.10 | 20 | 15% | 50 | 40 |

## Model Sizes

| Model | Hidden | Layers | FFN | ~Params |
|-------|--------|--------|-----|---------|
| base | 384 | 5 | 1536 | 9.4M |
| medium | 512 | 6 | 2048 | ~18M |
| large | 512 | 8 | 2048 | ~24M |

## Running

```bash
# Phase 1: Light CR-CTC on base model
python scripts/train_model.py experiment=transformer/crctc_sweep/base_ctrl
python scripts/train_model.py experiment=transformer/crctc_sweep/base_b05_m10_s50
python scripts/train_model.py experiment=transformer/crctc_sweep/base_b10_m20_s50
python scripts/train_model.py experiment=transformer/crctc_sweep/base_b15_m30_s40

# Phase 2: Scale up
python scripts/train_model.py experiment=transformer/crctc_sweep/med_ctrl
python scripts/train_model.py experiment=transformer/crctc_sweep/med_b05_m10_s50
python scripts/train_model.py experiment=transformer/crctc_sweep/lg_ctrl
python scripts/train_model.py experiment=transformer/crctc_sweep/lg_b05_m10_s50
```
