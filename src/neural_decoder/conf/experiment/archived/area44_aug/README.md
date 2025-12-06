# Area 44 Augmentation Experiments

**Status: Archived** - These experiments explored using Area 44 as a noise source for data augmentation. The approach showed no improvement over baseline augmentation.

## Key Findings

1. **Area 44 has zero correlation with Area 6v** - Correlation analysis showed max |correlation| < 0.05
2. **CrossAreaMixup and AdaptiveNoise44** showed no improvement over white noise
3. **CorrelationGuidedDropout** was not viable due to zero correlations

## Active Config

- `baseline.yaml` - Control experiment (6v-only with standard white noise)

## Archived Experiments

Detailed augmentation experiments are in `archived/area44_aug/`:
- Mixup variants (01/02/03/05, calibrated, no-white)
- Adaptive noise variants (low/high, calibrated)
- Correlation dropout experiments
- Combined strategies

See the original experiment results in the project's experimental logs.
