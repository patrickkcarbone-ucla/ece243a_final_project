"""
Pre-Patch GMVAE TCP Trainer.

Combines:
- TCP multi-level loss (mono, diphone, context, fused, gate entropy)
- Frame-level GMVAE auxiliary loss (reconstruction, KL, categorical)

Key differences from dual_stream_trainer:
- No fusion gate loss (cluster probs are just extra input channels)
- GMVAE losses computed at frame level (before patching)
- Simpler loss combination

Loss function:
    L_total = L_fused                           # Primary TCP fused CTC
            + alpha_mono * L_mono               # TCP monophone CTC
            + alpha_di * L_diphone              # TCP diphone CTC
            + alpha_context * L_context         # TCP context CTC
            - lambda_gate * H(tcp_gates)        # TCP gate entropy
            + alpha_gmvae_rec * L_rec           # Frame reconstruction
            + alpha_gmvae_kl * L_kl             # GMVAE KL divergence
            + alpha_gmvae_cat * L_cat           # GMVAE categorical entropy
            + alpha_gmvae_sup * L_sup           # Supervised clustering (optional)
"""

import os
import time
import math
import csv
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from edit_distance import SequenceMatcher
from neural_decoder.trainers.base_trainer import BaseTrainer
from neural_decoder.schedulers import (
    create_phased_scheduler,
    create_loss_scheduler,
)
from neural_decoder.losses.gmvae_losses import GMVAELossFunctions


class GMVAETCPTrainer(BaseTrainer):
    """
    Trainer for Pre-Patch GMVAE TCP Transformer.

    Combines TCP multi-level losses with frame-level GMVAE auxiliary losses.
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

        # CTC loss
        self.loss_ctc = torch.nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True
        )

        # GMVAE loss functions
        self.gmvae_losses = GMVAELossFunctions()

        # Detect if using dual-region dataset
        sample_batch = next(iter(train_loader))
        self.dual_region = len(sample_batch) == 6

        # Detect if model uses TCP
        self.use_tcp = getattr(model, "use_tcp", False)

        # TCP loss weights
        self.alpha_fused = getattr(cfg, "alpha_fused", 1.0)
        self.alpha_mono = getattr(cfg, "alpha_mono", 0.3)
        self.alpha_di = getattr(cfg, "alpha_di", 0.4)
        self.alpha_context = getattr(cfg, "alpha_context", 0.3)
        self.lambda_gate = getattr(cfg, "lambda_gate", 0.01)

        # GMVAE loss weights (frame-level)
        self.alpha_gmvae_rec = getattr(cfg, "alpha_gmvae_rec", 0.1)
        self.alpha_gmvae_kl = getattr(cfg, "alpha_gmvae_kl", 0.5)
        self.alpha_gmvae_cat = getattr(cfg, "alpha_gmvae_cat", 10.0)
        self.alpha_gmvae_sup = getattr(cfg, "alpha_gmvae_sup", 0.0)
        self.gmvae_rec_type = getattr(cfg, "gmvae_rec_type", "mse")

        # Mixed precision
        self.use_amp = getattr(cfg, "useMixedPrecision", False)
        if self.use_amp:
            self.scaler = GradScaler()

        # LR warmup
        self.warmup_epochs = getattr(cfg, "lrWarmupEpochs", 5)

        # Load diphone vocab for TCP diphone loss
        self.diphone_vocab = None
        diphone_vocab_path = getattr(
            cfg, "diphone_vocab_path", "data/diphone_vocab.pkl"
        )
        if not os.path.isabs(diphone_vocab_path):
            try:
                import hydra

                orig_cwd = hydra.utils.get_original_cwd()
                diphone_vocab_path = os.path.join(orig_cwd, diphone_vocab_path)
            except Exception:
                pass

        if os.path.exists(diphone_vocab_path):
            with open(diphone_vocab_path, "rb") as f:
                self.diphone_vocab = pickle.load(f)
            print(
                f"Loaded diphone vocab: {len(self.diphone_vocab['diphone_to_idx'])} diphones"
            )
        else:
            print(f"WARNING: Diphone vocab not found at {diphone_vocab_path}")

        # Running statistics
        self._tcp_gate_sum = None
        self._tcp_gate_count = 0
        self._cluster_usage = None

        # Gradient flow tracking (to see if TCP uses GMVAE features)
        self._gmvae_grad_sum = 0.0
        self._gmvae_grad_count = 0

    def _unpack_batch(self, batch):
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
        return ((X_len - self.model.kernelLen) / self.model.strideLen).to(
            torch.int32
        )

    def _phone_to_diphone_labels(self, y, y_len):
        """Convert phoneme labels to diphone labels."""
        if self.diphone_vocab is None:
            raise RuntimeError("Diphone vocab not loaded")

        diphone_to_idx = self.diphone_vocab["diphone_to_idx"]
        B = y.shape[0]
        max_len = y.shape[1]

        diphone_y = torch.zeros(
            B, max_len - 1, dtype=torch.int32, device=y.device
        )
        diphone_len = torch.zeros(B, dtype=torch.int32, device=y.device)

        for b in range(B):
            seq_len = y_len[b].item()
            if seq_len < 2:
                continue

            for i in range(seq_len - 1):
                p1 = y[b, i].item()
                p2 = y[b, i + 1].item()
                diphone = (p1, p2)
                diphone_idx = diphone_to_idx.get(diphone, 0)
                diphone_y[b, i] = diphone_idx

            diphone_len[b] = seq_len - 1

        return diphone_y, diphone_len

    def _reset_epoch_stats(self):
        if self.use_tcp and getattr(self.model, "use_gating", False):
            self._tcp_gate_sum = torch.zeros(3, device=self.device)
            self._tcp_gate_count = 0
        else:
            self._tcp_gate_sum = None
            self._tcp_gate_count = 0

        self._cluster_usage = None

        # Reset gradient tracking
        self._gmvae_grad_sum = 0.0
        self._gmvae_grad_count = 0

        # Loss component tracking
        self._loss_components = {
            "L_fused": 0.0,
            "L_mono": 0.0,
            "L_di": 0.0,
            "L_context": 0.0,
            "tcp_gate_entropy": 0.0,
            "L_rec": 0.0,
            "L_kl": 0.0,
            "L_cat": 0.0,
            "L_sup": 0.0,
        }
        self._loss_count = 0

    def _update_epoch_stats(self, model_output):
        # TCP gate stats
        tcp_gates = model_output.get("gates")
        if tcp_gates is not None and self._tcp_gate_sum is not None:
            with torch.no_grad():
                batch_mean = tcp_gates.mean(dim=0)
                self._tcp_gate_sum += batch_mean
                self._tcp_gate_count += 1

        # Cluster usage
        gmvae_out = model_output.get("gmvae_output", {})
        prob_cat = gmvae_out.get("prob_cat")
        if prob_cat is not None:
            avg_prob = prob_cat.mean(dim=[0, 1]).detach().cpu()
            if self._cluster_usage is None:
                self._cluster_usage = avg_prob
            else:
                self._cluster_usage = (
                    0.9 * self._cluster_usage + 0.1 * avg_prob
                )

    def _get_epoch_stats_str(self):
        parts = []

        # TCP gates
        if self._tcp_gate_sum is not None and self._tcp_gate_count > 0:
            mean_gates = (
                self._tcp_gate_sum / float(self._tcp_gate_count)
            ).tolist()
            if len(mean_gates) >= 3:
                parts.append(
                    f"tcp_g=[{mean_gates[0]:.2f},{mean_gates[1]:.2f},{mean_gates[2]:.2f}]"
                )

        # Cluster entropy
        if self._cluster_usage is not None:
            p = self._cluster_usage
            entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
            max_entropy = math.log(len(p))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            parts.append(f"clust_H={norm_entropy:.2f}")

        return " ".join(parts)

    def _get_tcp_gate_avg(self):
        """Get average TCP gate weights as list [mono, di, ctx]."""
        if self._tcp_gate_sum is not None and self._tcp_gate_count > 0:
            return (self._tcp_gate_sum / float(self._tcp_gate_count)).tolist()
        return [0.0, 0.0, 0.0]

    def _get_cluster_entropy(self):
        """Get normalized cluster entropy."""
        if self._cluster_usage is not None:
            p = self._cluster_usage
            entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
            max_entropy = math.log(len(p))
            return entropy / max_entropy if max_entropy > 0 else 0
        return 0.0

    def save_stats_comprehensive(
        self,
        epoch,
        train_loss,
        train_per,
        test_loss,
        test_per,
        lr,
        loss_components,
        alpha_fused,
        alpha_mono,
        alpha_di,
        alpha_context,
        lambda_gate,
        alpha_gmvae_rec,
        alpha_gmvae_kl,
        alpha_gmvae_cat,
        alpha_gmvae_sup,
        gmvae_temp,
        tcp_gates,
        cluster_entropy,
        gmvae_grad_norm=0.0,
    ):
        """Save comprehensive training stats to CSV."""
        # Update base class stats
        self.trainLoss.append(train_loss)
        self.trainCER.append(train_per)
        self.testLoss.append(test_loss)
        self.testCER.append(test_per)

        # Build row data
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_per": train_per,
            "test_loss": test_loss,
            "test_per": test_per,
            "lr": lr,
            # Loss weights
            "alpha_fused": alpha_fused,
            "alpha_mono": alpha_mono,
            "alpha_di": alpha_di,
            "alpha_context": alpha_context,
            "lambda_gate": lambda_gate,
            "alpha_gmvae_rec": alpha_gmvae_rec,
            "alpha_gmvae_kl": alpha_gmvae_kl,
            "alpha_gmvae_cat": alpha_gmvae_cat,
            "alpha_gmvae_sup": alpha_gmvae_sup,
            # Loss components
            "L_fused": loss_components.get("L_fused", 0),
            "L_mono": loss_components.get("L_mono", 0),
            "L_di": loss_components.get("L_di", 0),
            "L_context": loss_components.get("L_context", 0),
            "tcp_gate_entropy": loss_components.get("tcp_gate_entropy", 0),
            "L_rec": loss_components.get("L_rec", 0),
            "L_kl": loss_components.get("L_kl", 0),
            "L_cat": loss_components.get("L_cat", 0),
            "L_sup": loss_components.get("L_sup", 0),
            # Model state
            "gmvae_temp": gmvae_temp,
            "tcp_gate_mono": tcp_gates[0] if len(tcp_gates) > 0 else 0,
            "tcp_gate_di": tcp_gates[1] if len(tcp_gates) > 1 else 0,
            "tcp_gate_ctx": tcp_gates[2] if len(tcp_gates) > 2 else 0,
            "cluster_entropy": cluster_entropy,
            "gmvae_grad_norm": gmvae_grad_norm,  # Gradient flow through GMVAE
        }

        # Initialize history on first call
        if not hasattr(self, "_stats_history"):
            self._stats_history = []
        self._stats_history.append(row)

        # Write CSV
        csv_path = os.path.join(self.cfg.outputDir, "training_stats.csv")
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerows(self._stats_history)
        except Exception as e:
            print(f"Warning: CSV save failed: {e}")

        # Save pickle for legacy compatibility
        tStats = {
            "trainLoss": np.array(self.trainLoss),
            "trainCER": np.array(self.trainCER),
            "testLoss": np.array(self.testLoss),
            "testCER": np.array(self.testCER),
        }
        try:
            with open(
                os.path.join(self.cfg.outputDir, "trainingStats"), "wb"
            ) as file:
                pickle.dump(tStats, file)
        except Exception as e:
            print(f"Warning: pickle save failed: {e}")

    def _ctc_loss(self, logits, targets, output_lens, target_lens):
        logits_f32 = logits.float()
        return self.loss_ctc(
            torch.permute(logits_f32.log_softmax(2), [1, 0, 2]),
            targets,
            output_lens,
            target_lens,
        )

    def _compute_tcp_losses(self, model_output, y, y_len, output_lens):
        """Compute TCP multi-level losses."""
        loss_dict = {}

        # Primary fused loss
        phone_logits = model_output.get("phone_logits")
        if phone_logits is not None:
            L_fused = self._ctc_loss(phone_logits, y, output_lens, y_len)
            L_fused = torch.sum(L_fused)
            loss_dict["L_fused"] = L_fused
        else:
            loss_dict["L_fused"] = torch.tensor(0.0, device=self.device)

        # Monophone loss
        mono_logits = model_output.get("mono_logits")
        if mono_logits is not None:
            L_mono = self._ctc_loss(mono_logits, y, output_lens, y_len)
            L_mono = torch.sum(L_mono)
            loss_dict["L_mono"] = L_mono
        else:
            loss_dict["L_mono"] = torch.tensor(0.0, device=self.device)

        # Diphone loss
        diphone_logits = model_output.get("diphone_logits")
        if diphone_logits is not None and self.diphone_vocab is not None:
            diphone_y, diphone_len = self._phone_to_diphone_labels(y, y_len)
            L_di = self._ctc_loss(
                diphone_logits, diphone_y, output_lens, diphone_len
            )
            L_di = torch.sum(L_di)
            loss_dict["L_di"] = L_di
        else:
            loss_dict["L_di"] = torch.tensor(0.0, device=self.device)

        # Context loss
        context_logits = model_output.get("context_logits")
        if context_logits is not None:
            L_context = self._ctc_loss(context_logits, y, output_lens, y_len)
            L_context = torch.sum(L_context)
            loss_dict["L_context"] = L_context
        else:
            loss_dict["L_context"] = torch.tensor(0.0, device=self.device)

        # TCP gate entropy regularization
        tcp_gates = model_output.get("gates")
        if tcp_gates is not None and self.lambda_gate > 0:
            gate_entropy = -torch.sum(
                tcp_gates * torch.log(tcp_gates + 1e-8), dim=-1
            ).mean()
            loss_dict["tcp_gate_entropy"] = gate_entropy
        else:
            loss_dict["tcp_gate_entropy"] = torch.tensor(
                0.0, device=self.device
            )

        return loss_dict

    def _compute_gmvae_losses(
        self, gmvae_output, neuralInput, y=None, y_len=None, output_lens=None
    ):
        """
        Compute GMVAE losses at frame level.

        The GMVAE operates on raw neural frames (20ms), so losses are computed
        at that resolution before patching.
        """
        loss_dict = {}

        # Frame reconstruction loss
        x_rec = gmvae_output.get("x_rec")
        if x_rec is not None:
            L_rec = self.gmvae_losses.reconstruction_loss(
                neuralInput, x_rec, self.gmvae_rec_type
            )
            loss_dict["L_rec"] = L_rec
        else:
            loss_dict["L_rec"] = torch.tensor(0.0, device=self.device)

        # KL divergence
        z = gmvae_output.get("z")
        mu = gmvae_output.get("mu")
        var = gmvae_output.get("var")
        y_mu = gmvae_output.get("y_mu")
        y_var = gmvae_output.get("y_var")

        if all(v is not None for v in [z, mu, var, y_mu, y_var]):
            L_kl = self.gmvae_losses.gaussian_kl_loss(z, mu, var, y_mu, y_var)
            loss_dict["L_kl"] = L_kl
        else:
            loss_dict["L_kl"] = torch.tensor(0.0, device=self.device)

        # Categorical entropy + diversity
        logits = gmvae_output.get("logits")
        prob_cat = gmvae_output.get("prob_cat")

        if logits is not None and prob_cat is not None:
            L_cat = self.gmvae_losses.categorical_entropy_loss(
                logits, prob_cat
            )
            L_div = self.gmvae_losses.cluster_diversity_loss(prob_cat)
            loss_dict["L_cat"] = L_cat + L_div
        else:
            loss_dict["L_cat"] = torch.tensor(0.0, device=self.device)

        # Supervised clustering loss (optional)
        # Note: This is trickier at frame level since we need frame-to-phoneme alignment
        # For now, we skip it (set alpha_gmvae_sup=0)
        if self.alpha_gmvae_sup > 0 and prob_cat is not None and y is not None:
            # We'd need to upsample labels to frame level or downsample prob_cat to patch level
            # For simplicity, we can pool prob_cat to patch level
            # This is a simplified version - proper implementation would need alignment
            B, T_frames, K = prob_cat.shape
            patch_len = self.model.patch_len
            patch_stride = self.model.patch_stride

            # Pool prob_cat to patch level (average over patch)
            num_patches = (T_frames - patch_len) // patch_stride + 1
            valid_len = (num_patches - 1) * patch_stride + patch_len
            prob_cat_valid = prob_cat[:, :valid_len, :]  # [B, valid_T, K]

            # Reshape and average pool
            prob_cat_patches = []
            for i in range(num_patches):
                start = i * patch_stride
                end = start + patch_len
                patch_prob = prob_cat_valid[:, start:end, :].mean(
                    dim=1
                )  # [B, K]
                prob_cat_patches.append(patch_prob)
            prob_cat_patched = torch.stack(
                prob_cat_patches, dim=1
            )  # [B, T', K]

            L_sup = self.gmvae_losses.supervised_cluster_loss(
                prob_cat_patched, y, output_lens, y_len
            )
            loss_dict["L_sup"] = L_sup
        else:
            loss_dict["L_sup"] = torch.tensor(0.0, device=self.device)

        return loss_dict

    def _compute_loss(self, model_output, neuralInput, y, y_len, output_lens):
        """Compute combined TCP + GMVAE loss."""
        loss_dict = {}

        # TCP losses
        tcp_loss_dict = self._compute_tcp_losses(
            model_output, y, y_len, output_lens
        )
        for k, v in tcp_loss_dict.items():
            loss_dict[k] = v.item() if torch.is_tensor(v) else v

        # GMVAE losses (frame level)
        gmvae_out = model_output.get("gmvae_output", {})
        gmvae_loss_dict = self._compute_gmvae_losses(
            gmvae_out, neuralInput, y, y_len, output_lens
        )
        for k, v in gmvae_loss_dict.items():
            loss_dict[k] = v.item() if torch.is_tensor(v) else v

        # Total loss
        total_loss = tcp_loss_dict["L_fused"]  # Primary loss

        # Add TCP aux losses
        if self.use_tcp:
            total_loss = total_loss + self.alpha_mono * tcp_loss_dict["L_mono"]
            total_loss = total_loss + self.alpha_di * tcp_loss_dict["L_di"]
            total_loss = (
                total_loss + self.alpha_context * tcp_loss_dict["L_context"]
            )
            total_loss = (
                total_loss
                - self.lambda_gate * tcp_loss_dict["tcp_gate_entropy"]
            )

        # Add GMVAE losses
        total_loss = (
            total_loss + self.alpha_gmvae_rec * gmvae_loss_dict["L_rec"]
        )
        total_loss = total_loss + self.alpha_gmvae_kl * gmvae_loss_dict["L_kl"]
        total_loss = (
            total_loss + self.alpha_gmvae_cat * gmvae_loss_dict["L_cat"]
        )
        total_loss = (
            total_loss + self.alpha_gmvae_sup * gmvae_loss_dict["L_sup"]
        )

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def _compute_gmvae_grad_norm(self):
        """Compute gradient norm for GMVAE parameters to check if TCP uses cluster features."""
        total_norm = 0.0
        count = 0

        # Check gradients on GMVAE's cluster inference network (q(y|x))
        if hasattr(self.model, "frame_gmvae"):
            for name, param in self.model.frame_gmvae.named_parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2
                    count += 1

        if count > 0:
            return (total_norm**0.5) / count  # Average norm per parameter
        return 0.0

    def _train_step(self, X, y, y_len, dayIdx, output_lens):
        if self.use_amp:
            with autocast():
                model_output = self.model.forward(X, dayIdx)
                loss, loss_dict = self._compute_loss(
                    model_output, X, y, y_len, output_lens
                )

            self.scaler.scale(loss).backward()

            # Track GMVAE gradient BEFORE clipping
            gmvae_grad = self._compute_gmvae_grad_norm()
            self._gmvae_grad_sum += gmvae_grad
            self._gmvae_grad_count += 1

            self.scaler.unscale_(self.optimizer)
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            model_output = self.model.forward(X, dayIdx)
            loss, loss_dict = self._compute_loss(
                model_output, X, y, y_len, output_lens
            )

            loss.backward()

            # Track GMVAE gradient BEFORE clipping
            gmvae_grad = self._compute_gmvae_grad_norm()
            self._gmvae_grad_sum += gmvae_grad
            self._gmvae_grad_count += 1

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

        with torch.no_grad():
            self._update_epoch_stats(model_output)
            self._accumulate_loss_components(loss_dict)

        return loss

    def _accumulate_loss_components(self, loss_dict):
        """Accumulate loss components for epoch averaging."""
        for key in self._loss_components:
            if key in loss_dict:
                self._loss_components[key] += loss_dict[key]
        self._loss_count += 1

    def _get_avg_loss_components(self):
        """Get average loss components for the epoch."""
        if self._loss_count == 0:
            return self._loss_components
        return {
            k: v / self._loss_count for k, v in self._loss_components.items()
        }

    def log_metrics_epoch(
        self,
        epoch,
        train_loss,
        train_per,
        test_loss,
        test_per,
        lr,
        loss_components=None,
        tcp_params=None,
        gmvae_params=None,
    ):
        endTime = time.time()
        elapsed = endTime - self.startTime

        # Line 1: Main metrics
        line1 = (
            f"epoch {epoch:3d}, "
            f"train_loss: {train_loss:7.4f}, train_per: {train_per:7.4f}, "
            f"test_loss: {test_loss:7.4f}, test_per: {test_per:7.4f}, "
            f"lr: {lr:8.6f}, time: {elapsed:7.1f}s"
        )
        print(line1.ljust(110))

        # Line 2: TCP losses + params
        tcp_line = None
        if loss_components is not None and self.use_tcp:
            tcp_losses = (
                f"L_fus={loss_components.get('L_fused', 0):7.3f} "
                f"L_mono={loss_components.get('L_mono', 0):7.3f} "
                f"L_di={loss_components.get('L_di', 0):7.3f} "
                f"L_ctx={loss_components.get('L_context', 0):7.3f} "
                f"H_tcp={loss_components.get('tcp_gate_entropy', 0):6.3f}"
            )
            tcp_params_str = ""
            if tcp_params:
                tcp_params_str = (
                    f" | α_m={tcp_params.get('alpha_mono', 0):5.2f} "
                    f"α_d={tcp_params.get('alpha_di', 0):5.2f} "
                    f"α_c={tcp_params.get('alpha_context', 0):5.2f} "
                    f"λ_g={tcp_params.get('lambda_gate', 0):6.3f} "
                    f"tcp_g={tcp_params.get('tcp_gates', '[?,?,?]')}"
                )
            tcp_line = f"    TCP: {tcp_losses}{tcp_params_str}"
            print(tcp_line.ljust(110))

        # Line 3: GMVAE losses + params
        gmvae_line = None
        if loss_components is not None:
            gmvae_losses = (
                f"L_rec={loss_components.get('L_rec', 0):7.3f} "
                f"L_kl={loss_components.get('L_kl', 0):7.3f} "
                f"L_cat={loss_components.get('L_cat', 0):7.3f} "
                f"L_sup={loss_components.get('L_sup', 0):7.3f}"
            )
            gmvae_params_str = ""
            if gmvae_params:
                gmvae_params_str = (
                    f" | α_rec={gmvae_params.get('alpha_rec', 0):6.3f} "
                    f"α_kl={gmvae_params.get('alpha_kl', 0):6.3f} "
                    f"α_cat={gmvae_params.get('alpha_cat', 0):5.2f} "
                    f"α_sup={gmvae_params.get('alpha_sup', 0):5.2f} "
                    f"temp={gmvae_params.get('temp', 0):5.3f} "
                    f"clust_H={gmvae_params.get('cluster_entropy', 0):5.2f} "
                    f"∇gmvae={gmvae_params.get('grad_norm', 0):6.4f}"
                )
            gmvae_line = f"    GMVAE: {gmvae_losses}{gmvae_params_str}"
            print(gmvae_line.ljust(110))

        try:
            from datetime import datetime

            timestamp = datetime.now().isoformat(timespec="seconds")
            with open(
                os.path.join(self.cfg.outputDir, "training_log.txt"), "a"
            ) as f:
                f.write(f"{timestamp} - {line1}\n")
                if tcp_line:
                    f.write(f"{timestamp} - {tcp_line}\n")
                if gmvae_line:
                    f.write(f"{timestamp} - {gmvae_line}\n")
        except Exception as e:
            print(f"Warning: log write failed: {e}")

        self.startTime = time.time()

    def fit(self):
        n_epochs = getattr(self.cfg, "nEpochs", 250)

        # LR schedule
        phased_scheduler = create_phased_scheduler(self.cfg, self.optimizer)
        use_phased_lr = phased_scheduler is not None

        if use_phased_lr:
            print(
                f"Using phased LR schedule with {len(phased_scheduler.phases)} phases"
            )

        # Loss schedule
        loss_defaults = {
            # TCP
            "alpha_fused": self.alpha_fused,
            "alpha_mono": self.alpha_mono,
            "alpha_di": self.alpha_di,
            "alpha_context": self.alpha_context,
            "lambda_gate": self.lambda_gate,
            # GMVAE
            "alpha_gmvae_rec": self.alpha_gmvae_rec,
            "alpha_gmvae_kl": self.alpha_gmvae_kl,
            "alpha_gmvae_cat": self.alpha_gmvae_cat,
            "alpha_gmvae_sup": self.alpha_gmvae_sup,
        }
        loss_scheduler = create_loss_scheduler(self.cfg, loss_defaults)
        use_loss_schedule = loss_scheduler is not None

        if use_loss_schedule:
            print(
                f"Using phased loss schedule for: {loss_scheduler.param_names}"
            )

        # Legacy LR decay
        decay_epoch = getattr(self.cfg, "lrDecayEpoch", 150)
        decay_factor = getattr(self.cfg, "lrDecayFactor", 0.1)
        decay_epoch_2 = getattr(self.cfg, "lrDecayEpoch2", None)
        decay_factor_2 = getattr(self.cfg, "lrDecayFactor2", 0.1)

        initial_lr = self.optimizer.param_groups[0]["lr"]
        current_lr = initial_lr
        lr_decayed = False
        lr_decayed_2 = False

        for epoch in range(n_epochs):
            self._reset_epoch_stats()

            # Update loss weights and GMVAE temperature
            if use_loss_schedule:
                loss_values = loss_scheduler.step(epoch)
                # TCP
                self.alpha_fused = loss_values.get(
                    "alpha_fused", self.alpha_fused
                )
                self.alpha_mono = loss_values.get(
                    "alpha_mono", self.alpha_mono
                )
                self.alpha_di = loss_values.get("alpha_di", self.alpha_di)
                self.alpha_context = loss_values.get(
                    "alpha_context", self.alpha_context
                )
                self.lambda_gate = loss_values.get(
                    "lambda_gate", self.lambda_gate
                )
                # GMVAE
                self.alpha_gmvae_rec = loss_values.get(
                    "alpha_gmvae_rec", self.alpha_gmvae_rec
                )
                self.alpha_gmvae_kl = loss_values.get(
                    "alpha_gmvae_kl", self.alpha_gmvae_kl
                )
                self.alpha_gmvae_cat = loss_values.get(
                    "alpha_gmvae_cat", self.alpha_gmvae_cat
                )
                self.alpha_gmvae_sup = loss_values.get(
                    "alpha_gmvae_sup", self.alpha_gmvae_sup
                )
                # GMVAE temperature
                if "gmvae_temp" in loss_values:
                    self.model.gmvae_temperature.fill_(
                        loss_values["gmvae_temp"]
                    )
                else:
                    self.model.update_gmvae_temperature(epoch)
            else:
                self.model.update_gmvae_temperature(epoch)

            # LR scheduling
            if use_phased_lr:
                current_lr = phased_scheduler.step(epoch)
            elif epoch < self.warmup_epochs:
                warmup_lr = initial_lr * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr
                current_lr = warmup_lr
            elif epoch == self.warmup_epochs:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = initial_lr
                current_lr = initial_lr
            else:
                if epoch == decay_epoch and not lr_decayed:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] *= decay_factor
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    lr_decayed = True
                elif (
                    decay_epoch_2 is not None
                    and epoch == decay_epoch_2
                    and not lr_decayed_2
                ):
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] *= decay_factor_2
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    lr_decayed_2 = True

            # Training epoch
            self.model.train()
            self.augmenter.train()

            epoch_losses = []
            for batch_data in self.train_loader:
                X, X_44, y, X_len, y_len, dayIdx = self._unpack_batch(
                    batch_data
                )

                self.optimizer.zero_grad()
                output_lens = self._compute_output_lens(X_len)

                X_aug = self.augmenter(X, X_44)
                loss = self._train_step(X_aug, y, y_len, dayIdx, output_lens)
                epoch_losses.append(loss.item())

            # Evaluation
            with torch.no_grad():
                self.model.eval()
                self.augmenter.eval()

                train_loss, train_per = self._evaluate_loader(
                    self.train_loader, max_batches=10
                )
                test_loss, test_per = self._evaluate_loader(self.test_loader)

            # Get average loss components
            avg_loss_components = self._get_avg_loss_components()

            # Build TCP params dict
            tcp_gates = self._get_tcp_gate_avg()
            tcp_gates_str = (
                f"[{tcp_gates[0]:.2f},{tcp_gates[1]:.2f},{tcp_gates[2]:.2f}]"
            )
            tcp_params = {
                "alpha_fused": self.alpha_fused,
                "alpha_mono": self.alpha_mono,
                "alpha_di": self.alpha_di,
                "alpha_context": self.alpha_context,
                "lambda_gate": self.lambda_gate,
                "tcp_gates": tcp_gates_str,
            }

            # Build GMVAE params dict
            gmvae_temp = self.model.gmvae_temperature.item()
            cluster_entropy = self._get_cluster_entropy()
            gmvae_grad_avg = self._gmvae_grad_sum / max(
                self._gmvae_grad_count, 1
            )
            gmvae_params = {
                "alpha_rec": self.alpha_gmvae_rec,
                "alpha_kl": self.alpha_gmvae_kl,
                "alpha_cat": self.alpha_gmvae_cat,
                "alpha_sup": self.alpha_gmvae_sup,
                "temp": gmvae_temp,
                "cluster_entropy": cluster_entropy,
                "grad_norm": gmvae_grad_avg,  # Shows if TCP backprops through GMVAE
            }

            self.log_metrics_epoch(
                epoch,
                train_loss,
                train_per,
                test_loss,
                test_per,
                current_lr,
                loss_components=avg_loss_components,
                tcp_params=tcp_params,
                gmvae_params=gmvae_params,
            )
            self.save_checkpoint(self.model, test_per)

            # Save comprehensive stats
            self.save_stats_comprehensive(
                epoch=epoch,
                train_loss=train_loss,
                train_per=train_per,
                test_loss=test_loss,
                test_per=test_per,
                lr=current_lr,
                loss_components=avg_loss_components,
                alpha_fused=self.alpha_fused,
                alpha_mono=self.alpha_mono,
                alpha_di=self.alpha_di,
                alpha_context=self.alpha_context,
                lambda_gate=self.lambda_gate,
                alpha_gmvae_rec=self.alpha_gmvae_rec,
                alpha_gmvae_kl=self.alpha_gmvae_kl,
                alpha_gmvae_cat=self.alpha_gmvae_cat,
                alpha_gmvae_sup=self.alpha_gmvae_sup,
                gmvae_temp=gmvae_temp,
                tcp_gates=tcp_gates,
                cluster_entropy=cluster_entropy,
                gmvae_grad_norm=gmvae_grad_avg,
            )

    def _evaluate_loader(self, loader, max_batches=None):
        all_loss = []
        total_edit_distance = 0
        total_seq_length = 0

        for batch_idx, batch_data in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            X, X_44, y, X_len, y_len, dayIdx = self._unpack_batch(batch_data)
            output_lens = self._compute_output_lens(X_len)

            if self.use_amp:
                with autocast():
                    model_output = self.model.forward(X, dayIdx)
            else:
                model_output = self.model.forward(X, dayIdx)

            pred = model_output["phone_logits"]
            pred_f32 = pred.float()

            loss = self.loss_ctc(
                torch.permute(pred_f32.log_softmax(2), [1, 0, 2]),
                y,
                output_lens,
                y_len,
            )
            loss = torch.sum(loss)
            all_loss.append(loss.cpu().detach().numpy())

            for iterIdx in range(pred.shape[0]):
                seq_len = output_lens[iterIdx].item()
                if seq_len <= 0:
                    continue

                decodedSeq = torch.argmax(pred[iterIdx, :seq_len, :], dim=-1)
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq.cpu().detach().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != 0])

                trueSeq = np.array(y[iterIdx][: y_len[iterIdx]].cpu().detach())

                matcher = SequenceMatcher(
                    a=trueSeq.tolist(), b=decodedSeq.tolist()
                )
                total_edit_distance += matcher.distance()
                total_seq_length += len(trueSeq)

        n_batches = (
            min(len(loader), max_batches) if max_batches else len(loader)
        )
        avg_loss = np.sum(all_loss) / max(n_batches, 1)
        per = (
            total_edit_distance / total_seq_length
            if total_seq_length > 0
            else 1.0
        )

        return avg_loss, per
