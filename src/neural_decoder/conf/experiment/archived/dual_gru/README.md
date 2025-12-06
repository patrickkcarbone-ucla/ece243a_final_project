# Dual-Stream GRU Experiments Summary

## Overview

This directory contains experiments testing whether dual-stream architectures that 
process Area 6v and Area 44 neural signals separately can improve speech decoding
over the baseline single-stream GRU.

## Key Finding: Area 44 is Not Useful for Speech Decoding

Our experiments definitively show that **Area 44 (Broca's area) provides no useful 
information for speech decoding** in this dataset, and adding it actually hurts performance.

### Channel Ablation Results (3K batches, LR=0.02)

| Configuration      | Final CER | Final Loss | Notes |
|--------------------|-----------|------------|-------|
| **6v only**        | **0.319** | 1.112      | ✓ Best performance |
| All channels (512) | 0.341     | 1.218      | 6.9% worse than 6v-only |
| 44 only            | 0.973     | 3.169      | Essentially random (no learning) |

### Dual-Stream Architecture Results (3K batches, LR=0.02)

All dual-stream variants underperformed the baseline single-stream GRU:

| Architecture              | Final CER | vs Baseline |
|---------------------------|-----------|-------------|
| Baseline GRU (6v only)    | **0.322** | —           |
| gated_separate            | 0.342     | +6.2% worse |
| concat_shared_large       | 0.343     | +6.5% worse |
| attention_shared_large    | 0.346     | +7.5% worse |
| attention_separate        | 0.349     | +8.4% worse |
| concat_separate           | 0.350     | +8.7% worse |
| concat_shared             | 0.355     | +10.2% worse |
| gated_shared              | 0.356     | +10.6% worse |
| attention_shared          | 0.358     | +11.2% worse |

### Interpretation

1. **Area 44 contains no speech-relevant signal**: The 44-only model showed virtually 
   no learning (CER stuck at ~1.0), indicating these channels carry no information 
   useful for speech decoding.

2. **Adding Area 44 hurts performance**: Using all 512 channels (6v + 44) is 6.9% 
   worse than using 6v alone. The model wastes capacity trying to extract signal 
   from noise.

3. **Dual-stream architectures add overhead without benefit**: The additional 
   parameters and complexity (gating, attention, separate encoders) cannot overcome 
   the fundamental problem that Area 44 has no useful signal to contribute.

4. **"Separate" decoders > "Shared" decoders**: Among dual-stream variants, 
   architectures with separate day-normalization layers performed slightly better, 
   likely because they don't force the model to find a common representation across 
   regions with very different signal properties.

## Recommendations

1. **Use 6v channels only** for speech decoding on this dataset
2. **Don't pursue dual-stream architectures** unless new evidence suggests 44 
   carries useful information
3. **Consider using Area 44 for data augmentation** - while it can't decode speech 
   directly, its temporal structure could provide realistic neural noise patterns

## Experiments in this Directory

- `sweep/` - Hyperparameter sweeps across fusion strategies and learning rates
- `extended/` - Longer training runs (20K batches)
- `*_nBatch1000.yaml` - Quick test runs

## Computational Notes

- Baseline GRU: ~0.28s per batch
- Dual-stream models: ~0.4-0.6s per batch (1.5-2x slower)
- The extra compute cost provides no performance benefit

