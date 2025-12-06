"""
Standalone GMVAE Trainer for Neural Speech Decoding.

Trains a GMVAE model independently (Phase 1) for later use as
a feature extractor for TCP models (Phase 2).

Features:
- Real-time logging with exact format specification
- Separate evaluation head for PER monitoring
- Support for both unsupervised (alpha_sup=0) and supervised modes
- Saves model weights and logs during training (not just after)

Loss function:
    L_total = alpha_rec * L_rec           # Reconstruction
            + alpha_kl * L_kl             # KL divergence
            + alpha_cat * L_cat           # Categorical entropy
            + alpha_sup * L_sup           # Supervised clustering (optional)

Logging format:
    epoch 149, train_loss:  0.3779, train_per:  0.1342, test_loss:  0.8578, test_per:  0.2219, lr: 0.000020, time:    96.5s
             GMVAE: L_rec=0.117 L_kl=0.000 L_cat=-0.944 L_sup=2.074 L_div=0.0006 | α_rec=1.000 α_kl=20.000 α_cat=1.000 α_sup=5.00 temp=1.000 clust_H=0.56
"""

import os
import time
import math
import csv
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from edit_distance import SequenceMatcher

from neural_decoder.trainers.base_trainer import BaseTrainer
from neural_decoder.schedulers import create_phased_scheduler, create_loss_scheduler
from neural_decoder.losses.gmvae_losses import GMVAELossFunctions


class GMVAEStandaloneTrainer(BaseTrainer):
    """
    Trainer for standalone GMVAE model.

    Trains GMVAE with reconstruction, KL, categorical, and optional
    supervised losses. Includes a separate evaluation head for PER
    monitoring that does not affect GMVAE training.
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
        super().__init__(cfg)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.augmenter = augmenter
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # Gradient clipping
        self.max_grad_norm = getattr(cfg, "maxGradNorm", 1.0)

        # CTC loss for evaluation head
        self.loss_ctc = torch.nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True
        )

        # GMVAE loss functions
        self.gmvae_losses = GMVAELossFunctions()

        # Detect if using dual-region dataset
        sample_batch = next(iter(train_loader))
        self.dual_region = len(sample_batch) == 6

        # GMVAE loss weights
        self.alpha_rec = getattr(cfg, "alpha_gmvae_rec", 1.0)
        self.alpha_kl = getattr(cfg, "alpha_gmvae_kl", 5.0)
        self.alpha_cat = getattr(cfg, "alpha_gmvae_cat", 0.0)
        self.alpha_sup = getattr(cfg, "alpha_gmvae_sup", 0.0)

        # Reconstruction type
        self.rec_type = getattr(cfg, "gmvae_rec_type", "mse")

        # Separate optimizer for evaluation head
        self.eval_head_lr = getattr(cfg, "eval_head_lr", 0.001)
        self.eval_head_optimizer = torch.optim.Adam(
            model.eval_head.parameters(), lr=self.eval_head_lr
        )

        # Training statistics
        self._loss_components = {}
        self._loss_count = 0
        self._cluster_usage = None

        # Log file path for real-time saving
        self.log_file_path = os.path.join(cfg.outputDir, "training_log.txt")
        self.csv_file_path = os.path.join(cfg.outputDir, "training_stats.csv")
        self._stats_history = []

        # Initialize log file
        self._init_log_file()

    def _init_log_file(self):
        """Initialize log file with header."""
        try:
            with open(self.log_file_path, "w") as f:
                f.write("=" * 120 + "\n")
                f.write("GMVAE Standalone Training Log\n")
                f.write(f"Output Directory: {self.cfg.outputDir}\n")
                f.write("=" * 120 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not initialize log file: {e}")

    def _unpack_batch(self, batch):
        """Unpack batch from dataloader."""
        if self.dual_region:
            X, X_44, y, X_len, y_len, dayIdx = batch
            X_44 = X_44.to(self.device)
        else:
            X, y, X_len, y_len, dayIdx = batch
            X_44 = None

        X = X.to(self.device)
        y = y.to(self.device)
        X_len = X_len.to(self.device)
        y_len = y_len.to(self.device)
        dayIdx = dayIdx.to(self.device)

        return X, X_44, y, X_len, y_len, dayIdx

    def _compute_output_lens(self, X_len):
        """Compute output sequence lengths after patching."""
        return ((X_len - self.model.kernelLen) / self.model.strideLen).to(
            torch.int32
        )

    def _reset_epoch_stats(self):
        """Reset statistics at the start of each epoch."""
        self._loss_components = {
            "L_rec": 0.0,
            "L_kl": 0.0,
            "L_cat": 0.0,
            "L_sup": 0.0,
            "L_div": 0.0,
        }
        self._loss_count = 0
        self._cluster_usage = None

    def _update_cluster_stats(self, prob_cat):
        """Update running cluster usage statistics."""
        with torch.no_grad():
            avg_prob = prob_cat.mean(dim=[0, 1]).detach().cpu()
            if self._cluster_usage is None:
                self._cluster_usage = avg_prob
            else:
                self._cluster_usage = 0.9 * self._cluster_usage + 0.1 * avg_prob

    def _get_cluster_entropy(self):
        """Get normalized cluster entropy (0 = all same, 1 = uniform)."""
        if self._cluster_usage is not None:
            p = self._cluster_usage
            entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
            max_entropy = math.log(len(p))
            return entropy / max_entropy if max_entropy > 0 else 0
        return 0.0

    def _compute_gmvae_losses(self, model_output, y=None, y_len=None, output_lens=None):
        """
        Compute GMVAE losses.

        Args:
            model_output: Output dict from model forward pass
            y: [B, L] phoneme labels (for supervised loss)
            y_len: [B] label lengths
            output_lens: [B] output sequence lengths

        Returns:
            loss_dict: Dictionary of individual losses
            total_loss: Weighted sum of losses
        """
        loss_dict = {}

        # Get model outputs
        patches = model_output['patches']
        x_rec = model_output['x_rec']
        prob_cat = model_output['prob_cat']
        logits = model_output['logits']
        z = model_output['z']
        mu = model_output['mu']
        var = model_output['var']
        y_mu = model_output['y_mu']
        y_var = model_output['y_var']

        # L_rec: Reconstruction loss
        L_rec = self.gmvae_losses.reconstruction_loss(patches, x_rec, self.rec_type)
        loss_dict['L_rec'] = L_rec

        # L_kl: KL divergence loss
        L_kl = self.gmvae_losses.gaussian_kl_loss(z, mu, var, y_mu, y_var)
        loss_dict['L_kl'] = L_kl

        # L_cat: Categorical entropy loss
        L_cat = self.gmvae_losses.categorical_entropy_loss(logits, prob_cat)
        loss_dict['L_cat'] = L_cat

        # L_div: Cluster diversity loss (encourage uniform cluster usage)
        L_div = self.gmvae_losses.cluster_diversity_loss(prob_cat)
        loss_dict['L_div'] = L_div

        # L_sup: Supervised clustering loss (optional)
        if self.alpha_sup > 0 and y is not None and output_lens is not None:
            L_sup = self.gmvae_losses.supervised_cluster_loss(
                prob_cat, y, output_lens, y_len
            )
            loss_dict['L_sup'] = L_sup
        else:
            loss_dict['L_sup'] = torch.tensor(0.0, device=self.device)

        # Compute total weighted loss
        total_loss = (
            self.alpha_rec * L_rec
            + self.alpha_kl * L_kl
            + self.alpha_cat * L_cat
            + self.alpha_sup * loss_dict['L_sup']
        )

        return loss_dict, total_loss

    def _train_eval_head(self, model_output, y, y_len, output_lens):
        """
        Train evaluation head separately (detached from GMVAE).

        Args:
            model_output: Output dict from model forward pass
            y: [B, L] phoneme labels
            y_len: [B] label lengths
            output_lens: [B] output sequence lengths

        Returns:
            eval_loss: CTC loss for evaluation head
        """
        # Detach cluster probs - no gradients to GMVAE
        prob_cat_detached = model_output['prob_cat'].detach()

        # Forward through evaluation head
        phone_logits = self.model.eval_head(prob_cat_detached)

        # Compute CTC loss
        phone_logits_f32 = phone_logits.float()
        eval_loss = self.loss_ctc(
            torch.permute(phone_logits_f32.log_softmax(2), [1, 0, 2]),
            y,
            output_lens,
            y_len,
        )

        # Backward on eval head only
        self.eval_head_optimizer.zero_grad()
        eval_loss.backward()
        self.eval_head_optimizer.step()

        return eval_loss.item()

    def _accumulate_loss_components(self, loss_dict):
        """Accumulate loss components for epoch averaging."""
        for key in self._loss_components:
            if key in loss_dict:
                val = loss_dict[key]
                if torch.is_tensor(val):
                    val = val.item()
                self._loss_components[key] += val
        self._loss_count += 1

    def _get_avg_loss_components(self):
        """Get average loss components for the epoch."""
        if self._loss_count == 0:
            return self._loss_components
        return {k: v / self._loss_count for k, v in self._loss_components.items()}

    def _compute_per(self, loader, max_batches=None):
        """
        Compute Phoneme Error Rate using evaluation head.

        Args:
            loader: DataLoader to evaluate
            max_batches: Maximum number of batches to evaluate

        Returns:
            avg_loss: Average GMVAE loss
            per: Phoneme Error Rate
        """
        self.model.eval()

        all_losses = []
        total_edit_distance = 0
        total_seq_length = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                X, X_44, y, X_len, y_len, dayIdx = self._unpack_batch(batch_data)
                output_lens = self._compute_output_lens(X_len)

                # Forward pass
                model_output = self.model(X, dayIdx)

                # Compute GMVAE loss
                loss_dict, total_loss = self._compute_gmvae_losses(
                    model_output, y, y_len, output_lens
                )
                all_losses.append(total_loss.item())

                # Compute PER using eval head
                phone_logits = model_output['phone_logits']

                for iterIdx in range(phone_logits.shape[0]):
                    seq_len = output_lens[iterIdx].item()
                    if seq_len <= 0:
                        continue

                    # CTC decode
                    decodedSeq = torch.argmax(
                        phone_logits[iterIdx, :seq_len, :], dim=-1
                    )
                    decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                    decodedSeq = decodedSeq.cpu().detach().numpy()
                    decodedSeq = np.array([i for i in decodedSeq if i != 0])

                    trueSeq = np.array(
                        y[iterIdx][: y_len[iterIdx]].cpu().detach()
                    )

                    # Edit distance
                    matcher = SequenceMatcher(
                        a=trueSeq.tolist(), b=decodedSeq.tolist()
                    )
                    total_edit_distance += matcher.distance()
                    total_seq_length += len(trueSeq)

        avg_loss = np.mean(all_losses) if all_losses else 0.0
        per = total_edit_distance / total_seq_length if total_seq_length > 0 else 1.0

        return avg_loss, per

    def _log_epoch(
        self,
        epoch,
        train_loss,
        train_per,
        test_loss,
        test_per,
        lr,
        loss_components,
        elapsed_time,
    ):
        """
        Log epoch metrics in exact format and save to file.

        Format:
            epoch 149, train_loss:  0.3779, train_per:  0.1342, test_loss:  0.8578, test_per:  0.2219, lr: 0.000020, time:    96.5s
                     GMVAE: L_rec=0.117 L_kl=0.000 L_cat=-0.944 L_sup=2.074 L_div=0.0006 | α_rec=1.000 α_kl=20.000 α_cat=1.000 α_sup=5.00 temp=1.000 clust_H=0.56
        """
        temp = self.model.temperature.item()
        clust_H = self._get_cluster_entropy()

        # Line 1: Main metrics
        line1 = (
            f"epoch {epoch:3d}, "
            f"train_loss: {train_loss:7.4f}, train_per: {train_per:7.4f}, "
            f"test_loss: {test_loss:7.4f}, test_per: {test_per:7.4f}, "
            f"lr: {lr:8.6f}, time: {elapsed_time:7.1f}s"
        )

        # Line 2: GMVAE losses and hyperparameters
        line2 = (
            f"         GMVAE: "
            f"L_rec={loss_components.get('L_rec', 0):5.3f} "
            f"L_kl={loss_components.get('L_kl', 0):5.3f} "
            f"L_cat={loss_components.get('L_cat', 0):6.3f} "
            f"L_sup={loss_components.get('L_sup', 0):5.3f} "
            f"L_div={loss_components.get('L_div', 0):6.4f} | "
            f"α_rec={self.alpha_rec:5.3f} "
            f"α_kl={self.alpha_kl:6.3f} "
            f"α_cat={self.alpha_cat:5.3f} "
            f"α_sup={self.alpha_sup:5.2f} "
            f"temp={temp:5.3f} "
            f"clust_H={clust_H:4.2f}"
        )

        # Print to console
        print(line1)
        print(line2)

        # Save to log file immediately
        try:
            with open(self.log_file_path, "a") as f:
                f.write(line1 + "\n")
                f.write(line2 + "\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")

    def _save_stats_csv(
        self,
        epoch,
        train_loss,
        train_per,
        test_loss,
        test_per,
        lr,
        loss_components,
    ):
        """Save comprehensive stats to CSV file."""
        temp = self.model.temperature.item()
        clust_H = self._get_cluster_entropy()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_per": train_per,
            "test_loss": test_loss,
            "test_per": test_per,
            "lr": lr,
            "L_rec": loss_components.get("L_rec", 0),
            "L_kl": loss_components.get("L_kl", 0),
            "L_cat": loss_components.get("L_cat", 0),
            "L_sup": loss_components.get("L_sup", 0),
            "L_div": loss_components.get("L_div", 0),
            "alpha_rec": self.alpha_rec,
            "alpha_kl": self.alpha_kl,
            "alpha_cat": self.alpha_cat,
            "alpha_sup": self.alpha_sup,
            "temp": temp,
            "clust_H": clust_H,
        }

        self._stats_history.append(row)

        # Write CSV (overwrite entire file each time for consistency)
        try:
            with open(self.csv_file_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerows(self._stats_history)
        except Exception as e:
            print(f"Warning: Could not save CSV: {e}")

        # Also save pickle for legacy compatibility
        tStats = {
            "trainLoss": np.array([r["train_loss"] for r in self._stats_history]),
            "trainCER": np.array([r["train_per"] for r in self._stats_history]),
            "testLoss": np.array([r["test_loss"] for r in self._stats_history]),
            "testCER": np.array([r["test_per"] for r in self._stats_history]),
        }
        try:
            with open(os.path.join(self.cfg.outputDir, "trainingStats"), "wb") as f:
                pickle.dump(tStats, f)
        except Exception as e:
            print(f"Warning: Could not save pickle stats: {e}")

    def _save_model_weights(self, epoch, test_per):
        """Save model weights if performance improved."""
        if not getattr(self.cfg, "saveModelWeights", True):
            return

        # Track best performance
        if not hasattr(self, "_best_per"):
            self._best_per = float("inf")

        if test_per < self._best_per:
            self._best_per = test_per
            weights_path = os.path.join(self.cfg.outputDir, "modelWeights")
            try:
                torch.save(self.model.state_dict(), weights_path)
                print(f"    -> Saved best model (PER={test_per:.4f})")
            except Exception as e:
                print(f"Warning: Could not save model weights: {e}")

        # Also save latest weights
        latest_path = os.path.join(self.cfg.outputDir, "modelWeights_latest")
        try:
            torch.save(self.model.state_dict(), latest_path)
        except Exception as e:
            print(f"Warning: Could not save latest weights: {e}")

    def fit(self):
        """Main training loop."""
        n_epochs = getattr(self.cfg, "nEpochs", 50)

        # LR schedule
        phased_scheduler = create_phased_scheduler(self.cfg, self.optimizer)
        use_phased_lr = phased_scheduler is not None

        if use_phased_lr:
            print(f"Using phased LR schedule with {len(phased_scheduler.phases)} phases")

        # Loss schedule
        loss_defaults = {
            "alpha_gmvae_rec": self.alpha_rec,
            "alpha_gmvae_kl": self.alpha_kl,
            "alpha_gmvae_cat": self.alpha_cat,
            "alpha_gmvae_sup": self.alpha_sup,
        }
        loss_scheduler = create_loss_scheduler(self.cfg, loss_defaults)
        use_loss_schedule = loss_scheduler is not None

        if use_loss_schedule:
            print(f"Using phased loss schedule for: {loss_scheduler.param_names}")

        # Initial LR
        current_lr = self.optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 120)
        print("Starting GMVAE Standalone Training")
        print(f"Epochs: {n_epochs}")
        print(f"Initial LR: {current_lr:.6f}")
        print(f"alpha_rec={self.alpha_rec}, alpha_kl={self.alpha_kl}, "
              f"alpha_cat={self.alpha_cat}, alpha_sup={self.alpha_sup}")
        print("=" * 120 + "\n")

        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            self._reset_epoch_stats()

            # Update loss weights from schedule
            if use_loss_schedule:
                loss_values = loss_scheduler.step(epoch)
                self.alpha_rec = loss_values.get("alpha_gmvae_rec", self.alpha_rec)
                self.alpha_kl = loss_values.get("alpha_gmvae_kl", self.alpha_kl)
                self.alpha_cat = loss_values.get("alpha_gmvae_cat", self.alpha_cat)
                self.alpha_sup = loss_values.get("alpha_gmvae_sup", self.alpha_sup)

                # Temperature scheduling
                if "gmvae_temp" in loss_values:
                    self.model.temperature.fill_(loss_values["gmvae_temp"])
                else:
                    self.model.update_temperature(epoch)
            else:
                self.model.update_temperature(epoch)

            # LR scheduling
            if use_phased_lr:
                current_lr = phased_scheduler.step(epoch)
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            # Training phase
            self.model.train()

            for batch_data in self.train_loader:
                X, X_44, y, X_len, y_len, dayIdx = self._unpack_batch(batch_data)
                output_lens = self._compute_output_lens(X_len)

                # Forward pass
                self.optimizer.zero_grad()
                model_output = self.model(X, dayIdx)

                # Compute GMVAE losses
                loss_dict, total_loss = self._compute_gmvae_losses(
                    model_output, y, y_len, output_lens
                )

                # Backward pass for GMVAE
                total_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()

                # Train evaluation head separately
                self._train_eval_head(model_output, y, y_len, output_lens)

                # Update statistics
                self._accumulate_loss_components(loss_dict)
                self._update_cluster_stats(model_output['prob_cat'])

            # Evaluation phase
            train_loss, train_per = self._compute_per(
                self.train_loader, max_batches=10
            )
            test_loss, test_per = self._compute_per(self.test_loader)

            # Get average loss components
            avg_loss_components = self._get_avg_loss_components()

            # Timing
            elapsed_time = time.time() - epoch_start_time

            # Log metrics (saves to file immediately)
            self._log_epoch(
                epoch,
                train_loss,
                train_per,
                test_loss,
                test_per,
                current_lr,
                avg_loss_components,
                elapsed_time,
            )

            # Save stats to CSV
            self._save_stats_csv(
                epoch,
                train_loss,
                train_per,
                test_loss,
                test_per,
                current_lr,
                avg_loss_components,
            )

            # Save model weights
            self._save_model_weights(epoch, test_per)

            # Update base class stats for compatibility
            self.trainLoss.append(train_loss)
            self.trainCER.append(train_per)
            self.testLoss.append(test_loss)
            self.testCER.append(test_per)

        print("\n" + "=" * 120)
        print("Training Complete!")
        print(f"Best Test PER: {self._best_per:.4f}")
        print(f"Weights saved to: {self.cfg.outputDir}")
        print("=" * 120)