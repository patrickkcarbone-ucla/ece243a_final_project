"""
TCP Transformer Trainer with Frozen GMVAE Features.

Extends TCPTransformerTrainer to handle models with frozen GMVAE.
The key difference is that only trainable (non-frozen) parameters
are passed to the optimizer.

This trainer is used for Phase 2 of the GMVAE→TCP pipeline:
1. Load pre-trained GMVAE weights (frozen)
2. Train TCP on neural data + cluster features
"""

import os
import torch
from neural_decoder.trainers.tcp_transformer_trainer import TCPTransformerTrainer


class TCPWithGMVAEFeaturesTrainer(TCPTransformerTrainer):
    """
    Trainer for TCP models with frozen GMVAE features.

    Inherits all functionality from TCPTransformerTrainer but:
    1. Recreates optimizer with only trainable parameters
    2. Logs additional cluster-related metrics
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        scheduler,
        augmenter,
        train_loader,
        test_loader,
        device,
    ):
        # First, recreate optimizer with only trainable parameters
        # The original optimizer includes frozen GMVAE params which is wasteful
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        frozen_params = [p for p in model.parameters() if not p.requires_grad]

        print(f"Model parameters:")
        print(f"  - Trainable: {sum(p.numel() for p in trainable_params):,}")
        print(f"  - Frozen (GMVAE): {sum(p.numel() for p in frozen_params):,}")

        # Recreate optimizer with only trainable params
        # Use same hyperparameters from the original optimizer
        lr = optimizer.param_groups[0]['lr']
        weight_decay = optimizer.param_groups[0].get('weight_decay', 0)

        # Determine optimizer type from config or default to AdamW
        optimizer_type = getattr(cfg, 'optimizer_type', 'adamw')
        if optimizer_type == 'adam':
            new_optimizer = torch.optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            new_optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
            )

        print(f"Recreated optimizer with {len(trainable_params)} parameter groups")

        # Call parent init with new optimizer
        super().__init__(
            cfg=cfg,
            model=model,
            optimizer=new_optimizer,
            scheduler=scheduler,
            augmenter=augmenter,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        # Store GMVAE info for logging
        self.gmvae_weights_path = getattr(cfg, 'gmvae_weights_path', 'unknown')

        # Log file path for real-time saving
        self.log_file_path = os.path.join(cfg.outputDir, "training_log.txt")
        self._init_log_file()

    def _init_log_file(self):
        """Initialize log file with header."""
        try:
            with open(self.log_file_path, "w") as f:
                f.write("=" * 120 + "\n")
                f.write("TCP with Frozen GMVAE Features - Training Log\n")
                f.write(f"Output Directory: {self.cfg.outputDir}\n")
                f.write(f"GMVAE Weights: {self.gmvae_weights_path}\n")
                f.write("=" * 120 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not initialize log file: {e}")

    def log_metrics_epoch(
        self, epoch, train_loss, train_per, test_loss, test_per, lr, extra=""
    ):
        """Override to also save to file immediately."""
        # Call parent logging
        super().log_metrics_epoch(
            epoch, train_loss, train_per, test_loss, test_per, lr, extra
        )

        # Also save to file immediately
        try:
            log_msg = (
                f"epoch {epoch}, "
                f"train_loss: {train_loss:>7.4f}, train_per: {train_per:>7.4f}, "
                f"test_loss: {test_loss:>7.4f}, test_per: {test_per:>7.4f}, "
                f"lr: {lr:.6f}"
            )
            if extra:
                log_msg += f" {extra}"

            with open(self.log_file_path, "a") as f:
                f.write(log_msg + "\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")