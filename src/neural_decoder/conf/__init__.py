"""
Hydra configuration package for the `neural_decoder` project.

This package is discovered by Hydra via:

    @hydra.main(config_path="conf", config_name="config")

in `neural_decoder.neural_decoder_trainer`.

Config entry point:
  - config.yaml

Config groups:
  - model/
      - gru/              Single-stream GRU variants (baseline)
          - base.yaml         Default configuration (256 channels)
          - dual_region.yaml  All 512 channels (dual-region data)
          - dual_region_6v_only.yaml   Area 6v only (dual-region data) ← BEST
          - dual_region_6v_h1536.yaml  1.5x hidden capacity
          - dual_region_6v_h2048.yaml  2x hidden capacity
          - dual_region_6v_l6.yaml     6 layers
      - transformer/      Time-Masked Transformer (epoch-based training)
          - base.yaml         Default configuration (384 hidden, 5 layers, 6 heads)
          - medium.yaml       Increased capacity (512 hidden, 6 layers, ~18M params)
          - large.yaml        Large capacity (512 hidden, 8 layers, ~24M params)
          - full.yaml         All Part 2 features (diphone + intermediate CTC + FP16)
          - tcp.yaml          TCP (Temporal Coarticulation Pyramid) with multi-level heads
      - dual_stream/      Dual-stream GRU with fusion (DEPRECATED - underperforms)
          
  - augmentations/
      - baseline.yaml              Standard single-region augmentation
      - dual_region_baseline.yaml  Standard dual-region augmentation
      - transformer_baseline.yaml  Transformer augmentation (lower noise)
      - white_sd10/12.yaml         White noise controls
      - exclusive_sd10/12.yaml     Stochastic exclusive mode
      - independent_sd10/12.yaml   Stochastic independent mode
      - archived/                  Old iterative experiments

  - optimizer/
      - adam.yaml         Default (lr=0.02)
      - adam_lr01.yaml    LR=0.01
      - adam_lr005.yaml   LR=0.005
      - adamw.yaml        AdamW for Transformer (lr=0.001, weight_decay=1e-5)

  - trainer/
      - ctc.yaml              CTC loss trainer (batch-based, for GRU)
      - transformer_ctc.yaml  CTC loss trainer (epoch-based, for Transformer)
      - tcp_ctc.yaml          TCP Transformer trainer (multi-level loss)

DietCORP test-time adaptation:
  - dietcorpEvalFrequency: N  Evaluate with DietCORP every N epochs (0=disabled, 50 recommended)
  - dietcorpViews: 64         Number of augmented views for adaptation
  - dietcorpLR: 0.0001        Learning rate for patch embedding adaptation

  - experiment/
      - baseline_gru/     Single-stream GRU experiments
      - dual_gru/         Dual-stream experiments (see README.md for results)
      - transformer/      Time-Masked Transformer experiments
          - base.yaml           Part 1: Core Transformer
          - full.yaml           Part 2: All enhancements (diphone + inter CTC + FP16)
          - intermediate_only   Part 2: Intermediate CTC only
          - fp16.yaml           Part 2: FP16 mixed precision only
          - crctc.yaml          Part 3: CR-CTC consistency regularization
          - quick/              100-epoch quick screen experiments
          - crctc_sweep/        CR-CTC + model capacity sweep
          - tcp/                TCP Transformer experiments
              - base.yaml         Fixed gate weights
              - gated.yaml        Learned gating network
              - curriculum.yaml   TCP curriculum schedule (α_di decay)
              - quick/            100-epoch quick screens
      - capacity_sweep/   Model capacity experiments
      - area44_aug/       Area 44 augmentation experiments (archived)
      - archived/         Old experiments preserved for reference

  - scripts/
      - inference_dietcorp.py  DietCORP test-time adaptation evaluation
"""
