# Archived Experiments

This directory contains experiments that are no longer actively maintained but 
preserved for reference and reproducibility.

## Contents

### `area44_aug/`
Experiments using Area 44 neural data as augmentation for Area 6v decoding.
**Conclusion**: No improvement over baseline. Area 44 shows zero correlation with 6v.

### `capacity_sweep/`
Systematic grid of model sizes × augmentation strategies.
Active baselines are in the main `capacity_sweep/` directory.

### `stochastic_aug/`
Stochastic augmentation modes (exclusive, independent) at different noise levels.
Experiments were paused before completion.

## Using Archived Configs

These configs can still be used via full path:
```bash
python scripts/train_model.py experiment=archived/area44_aug/mixup_02
```

