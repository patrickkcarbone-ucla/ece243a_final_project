# Archived Augmentation Configs

These configs were developed during Area 44 augmentation experiments.
They are preserved for reference but no longer actively maintained.

## Categories

### Mixup Variants
- `mixup_01/02/03/05.yaml` - CrossAreaMixup at different ratios
- `mixup_02_calibrated_*.yaml` - Calibrated noise levels
- `mixup_*_no_white.yaml` - Mixup as white noise replacement

### Adaptive Noise Variants  
- `adaptive_noise_*.yaml` - AdaptiveNoise44 at different strengths
- `adaptive_calibrated_*.yaml` - Calibrated noise levels
- `adaptive_only.yaml` - Adaptive as white noise replacement

### Correlation-Based (Non-Viable)
- `corr_dropout_*.yaml` - CorrelationGuidedDropout (zero correlations found)
- `*_corr*.yaml` - Combinations with correlation dropout

### Combinations
- `combo_sd12.yaml` - Mixup + Adaptive combined
- `all_three.yaml` - All strategies together

## Note

These configs use `DualRegionAugmentationPipeline` and require:
- `useDualRegion: true` in experiment config
- 512-channel dual-region dataset

