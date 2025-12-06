"""
TCP (Temporal Coarticulation Pyramid) Transformer Trainer.

Epoch-based CTC trainer for TCP Transformer models with multi-level loss:
- L_fused: Primary CTC loss on gated fusion output
- L_mono: Intermediate CTC loss from monophone head (Level 1)
- L_di: Diphone CTC loss from diphone head (Level 2)
- L_context: CTC loss from context head (Level 3)
- Gate entropy regularization to prevent gate collapse

Supports:
- TCP curriculum schedule (α_di: 0.5 → 0.3 over training)
- Feghhi diphone schedule (start 100% diphone, ramp up phoneme)
- Mixed precision (FP16) training
- Flexible phased LR schedules (warmup, hold, cosine, step, exponential)
"""

import os
import time
import copy
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


class TCPTransformerTrainer(BaseTrainer):
    """
    CTC Trainer for TCP Transformer models with multi-level loss computation.

    Supports:
    - Multi-level CTC losses (fused, mono, diphone, context)
    - TCP curriculum schedule for α_di
    - Feghhi diphone schedule
    - Gate entropy regularization
    - Mixed precision (FP16) training
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

        # Detect if using dual-region dataset
        sample_batch = next(iter(train_loader))
        self.dual_region = len(sample_batch) == 6

        # Detect if model uses TCP
        self.use_tcp = getattr(model, "use_tcp", False)

        # TCP loss weights
        self.alpha_fused = getattr(cfg, "alpha_fused", 1.0)  # Weight for primary fused loss
        self.alpha_mono = getattr(cfg, "alpha_mono", 0.3)
        self.alpha_di = getattr(cfg, "alpha_di", 0.4)
        self.alpha_context = getattr(cfg, "alpha_context", 0.3)
        self.lambda_gate = getattr(cfg, "lambda_gate", 0.01)

        # TCP curriculum schedule (α_di decay)
        self.use_tcp_curriculum = getattr(cfg, "use_tcp_curriculum", False)
        self.tcp_curriculum_start = getattr(cfg, "tcp_curriculum_start", 0.5)
        self.tcp_curriculum_end = getattr(cfg, "tcp_curriculum_end", 0.3)
        self.tcp_curriculum_epochs = getattr(cfg, "tcp_curriculum_epochs", 150)
        # Step-based decay parameters (set step_epochs > 0 to enable)
        self.tcp_curriculum_step_epochs = getattr(
            cfg, "tcp_curriculum_step_epochs", 0
        )
        self.tcp_curriculum_step_size = getattr(
            cfg, "tcp_curriculum_step_size", 0.1
        )
        self.tcp_curriculum_start_epoch = getattr(
            cfg, "tcp_curriculum_start_epoch", 0
        )

        # Feghhi diphone schedule (alternative to TCP curriculum)
        self.diphone_feghhi_schedule = getattr(
            cfg, "diphone_feghhi_schedule", False
        )
        self.diphone_warmup_epochs = getattr(cfg, "diphone_warmup_epochs", 10)
        self.diphone_ramp_epochs = getattr(cfg, "diphone_ramp_epochs", 25)
        self.diphone_max_alpha = getattr(cfg, "diphone_max_alpha", 0.6)

        # Mixed precision
        self.use_amp = getattr(cfg, "useMixedPrecision", False)
        if self.use_amp:
            self.scaler = GradScaler()

        # LR warmup
        self.warmup_epochs = getattr(cfg, "lrWarmupEpochs", 5)

        # DietCORP test-time adaptation (periodic evaluation)
        self.dietcorp_eval_frequency = getattr(cfg, "dietcorpEvalFrequency", 0)
        self.dietcorp_lr = getattr(cfg, "dietcorpLR", 0.0001)
        self.dietcorp_views = getattr(cfg, "dietcorpViews", 64)
        self.dietcorp_confidence_threshold = getattr(
            cfg, "dietcorpConfidenceThreshold", 0.5
        )

        # Load diphone vocab for diphone target generation
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

        # Running gate statistics for debugging (mono / diphone / context)
        self._gate_sum = None
        self._gate_count = 0

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
        """Compute output sequence lengths for CTC loss."""
        return ((X_len - self.model.kernelLen) / self.model.strideLen).to(
            torch.int32
        )

    # -------------------------------------------------------------------------
    # Gate statistics (for inspecting learned TCP gating behavior)
    # -------------------------------------------------------------------------
    def _reset_gate_stats(self):
        """Reset running gate statistics for the current epoch."""
        if not (self.use_tcp and getattr(self.model, "use_gating", False)):
            self._gate_sum = None
            self._gate_count = 0
            return

        self._gate_sum = torch.zeros(3, device=self.device)
        self._gate_count = 0

    def _update_gate_stats(self, gates):
        """
        Accumulate mean gate values for debugging/logging.

        Args:
            gates: [B, 3] tensor with (g_mono, g_di, g_context)
        """
        if gates is None or self._gate_sum is None:
            return

        with torch.no_grad():
            # Average over batch, keep per-level means
            batch_mean = gates.mean(dim=0)  # [3]
            self._gate_sum += batch_mean
            self._gate_count += 1

    def _get_gate_stats(self):
        """
        Return average gate values over the current epoch.

        Returns:
            (g_mono, g_di, g_context) tuple of floats, or None if unavailable.
        """
        if self._gate_sum is None or self._gate_count == 0:
            return None

        mean_gates = (self._gate_sum / float(self._gate_count)).tolist()
        if len(mean_gates) >= 3:
            return mean_gates[0], mean_gates[1], mean_gates[2]
        return None

    def _phone_to_diphone_labels(self, y, y_len):
        """
        Convert phoneme labels to diphone labels.

        Diphones are consecutive phoneme pairs. For sequence [a, b, c],
        diphones are [(a,b), (b,c)].

        Returns:
            diphone_y: [B, max_len-1] diphone indices
            diphone_len: [B] diphone sequence lengths
        """
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

    def _apply_time_masking(self, x, num_masks=20, max_mask_fraction=0.075):
        """
        Apply time-masking augmentation to input tensor for DietCORP.

        Uses zeros for masking since this operates on raw input (neural_dim),
        not the model's hidden space where the learnable mask token lives.
        """
        B, T, C = x.shape
        x_masked = x.clone()

        for _ in range(num_masks):
            mask_len = int(T * max_mask_fraction * torch.rand(1).item())
            if mask_len == 0:
                continue

            for b in range(B):
                start = torch.randint(0, max(1, T - mask_len), (1,)).item()
                end = min(start + mask_len, T)
                x_masked[b, start:end, :] = 0.0

        return x_masked

    def _adapt_and_infer_dietcorp(self, X, dayIdx, continuous=True):
        """
        Single-trial DietCORP adaptation and inference for TCP models (Feghhi et al. 2025).

        Implements DietCORP with confidence-based filtering:
        1. Generate pseudo-label by greedy decoding the model's output
        2. Check confidence - only adapt if average confidence > threshold
        3. Create Z augmented copies with white noise, baseline shift, time-masking
        4. Compute CTC loss between augmented predictions and pseudo-label
        5. Single gradient step with gradient clipping (0.5)
        6. Only update patch embedding layer

        Args:
            X: [1, T, C] single trial input
            dayIdx: [1] day index for the trial
            continuous: If True (paper default), don't reset weights between trials.

        Returns:
            final_output: Model output after adaptation
        """
        # Confidence threshold for adaptation
        confidence_threshold = getattr(
            self, "dietcorp_confidence_threshold", 0.5
        )

        # Save original state only if not continuous
        if not continuous:
            orig_state = copy.deepcopy(self.model.patch_embed.state_dict())

        # Step 1: Get pseudo-label by greedy decoding (eval mode, no augmentation)
        self.model.eval()
        with torch.no_grad():
            unadapted_out = self.model.forward(X, dayIdx)
            if isinstance(unadapted_out, dict):
                pseudo_logits = unadapted_out["phone_logits"]  # [1, T, C]
            else:
                pseudo_logits = unadapted_out

            # Compute confidence: max softmax probability at each time step
            pseudo_probs = F.softmax(pseudo_logits, dim=-1)  # [1, T, C]
            max_probs, _ = pseudo_probs.max(dim=-1)  # [1, T]
            avg_confidence = max_probs.mean().item()

            # Track confidence statistics
            if not hasattr(self, "_dietcorp_confidence_stats"):
                self._dietcorp_confidence_stats = {
                    "adapted": 0,
                    "skipped": 0,
                    "total_conf": 0.0,
                }
            self._dietcorp_confidence_stats["total_conf"] += avg_confidence

            # Skip adaptation for low-confidence predictions
            if avg_confidence < confidence_threshold:
                self._dietcorp_confidence_stats["skipped"] += 1
                for param in self.model.parameters():
                    param.requires_grad = True
                return unadapted_out

            self._dietcorp_confidence_stats["adapted"] += 1

            # Greedy decode: get the most likely phoneme at each time step
            raw_sequence = pseudo_logits.argmax(dim=-1).squeeze(0)  # [T]

            # CTC collapse: remove blanks (class 0) and consecutive duplicates
            collapsed = []
            prev = -1
            for idx in raw_sequence.tolist():
                if idx != 0 and idx != prev:
                    collapsed.append(idx)
                prev = idx

            if len(collapsed) == 0:
                collapsed = [1]

            pseudo_sequence = torch.tensor(
                collapsed, dtype=torch.long, device=X.device
            )
            pseudo_length = len(collapsed)

        # Step 2: Create Z augmented views using SAME augmentations as training
        views = []
        self.augmenter.train()
        for _ in range(self.dietcorp_views):
            aug_X = self.augmenter(X.clone(), None)
            views.append(aug_X)

        views_batch = torch.cat(views, dim=0)
        dayIdx_expanded = dayIdx.expand(self.dietcorp_views)

        # Freeze all parameters except patch embedding
        for name, param in self.model.named_parameters():
            param.requires_grad = "patch_embed" in name

        # Put model in TRAIN mode so it applies time-masking
        self.model.train()

        # Forward pass on augmented views
        if self.use_amp:
            with autocast():
                adapted_out = self.model.forward(views_batch, dayIdx_expanded)
        else:
            adapted_out = self.model.forward(views_batch, dayIdx_expanded)

        if isinstance(adapted_out, dict):
            adapted_logits = adapted_out["phone_logits"]
        else:
            adapted_logits = adapted_out

        # Step 3: Compute CTC loss between augmented predictions and pseudo-label
        pseudo_targets = pseudo_sequence.unsqueeze(0).expand(
            self.dietcorp_views, -1
        )  # [Z, L]

        log_probs = F.log_softmax(adapted_logits.float(), dim=-1)
        log_probs = log_probs.permute(1, 0, 2)  # [T, Z, C]

        T = log_probs.shape[0]
        input_lengths = torch.full(
            (self.dietcorp_views,), T, dtype=torch.long, device=X.device
        )
        target_lengths = torch.full(
            (self.dietcorp_views,),
            pseudo_length,
            dtype=torch.long,
            device=X.device,
        )

        adapt_loss = F.ctc_loss(
            log_probs,
            pseudo_targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )

        # Step 4: Single gradient step with clipping
        adapt_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                p
                for n, p in self.model.named_parameters()
                if "patch_embed" in n
            ],
            max_norm=0.5,
        )

        # DEBUG: Print on first trial only
        if not hasattr(self, "_dietcorp_debug_printed"):
            patch_params = [
                (n, p)
                for n, p in self.model.named_parameters()
                if "patch_embed" in n
            ]
            print(f"  [DietCORP] CTC Loss: {adapt_loss.item():.4f}")
            for n, p in patch_params:
                if p.grad is not None:
                    print(
                        f"  [DietCORP] {n}: grad_norm={p.grad.norm().item():.4f} (after clip)"
                    )
            self._dietcorp_debug_printed = True

        # Apply gradient update (paper LR = 5e-4)
        # DEBUG: Check weight change
        weight_before = self.model.patch_embed.weight.data.clone()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "patch_embed" in name and param.grad is not None:
                    # Must use .data.sub_() for in-place modification!
                    param.data.sub_(self.dietcorp_lr * param.grad)

        # DEBUG: Measure weight change
        weight_after = self.model.patch_embed.weight.data
        weight_diff_mean = (weight_after - weight_before).abs().mean().item()
        weight_diff_max = (weight_after - weight_before).abs().max().item()
        if not hasattr(self, "_dietcorp_weight_debug_printed"):
            print(
                f"  [DietCORP] Weight change: mean={weight_diff_mean:.6f}, max={weight_diff_max:.6f}"
            )
            self._dietcorp_weight_debug_printed = True

        self.model.zero_grad()

        # Inference with adapted model
        self.model.eval()
        self.augmenter.eval()
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    final_out = self.model.forward(X, dayIdx)
            else:
                final_out = self.model.forward(X, dayIdx)

        # Reset weights only if not continuous
        if not continuous:
            self.model.patch_embed.load_state_dict(orig_state)

        # Re-enable gradients on all parameters for subsequent training
        for param in self.model.parameters():
            param.requires_grad = True

        return final_out

    def _update_tcp_alpha_di(self, epoch):
        """
        Update α_di based on TCP curriculum schedule.

        Supports two modes:
        1. Linear interpolation (default): tcp_curriculum_start → tcp_curriculum_end over tcp_curriculum_epochs
        2. Step decay: Decrease by tcp_curriculum_step_size every tcp_curriculum_step_epochs
        """
        if not self.use_tcp_curriculum:
            return

        # Check for step-based decay mode
        if self.tcp_curriculum_step_epochs > 0:
            # Step decay: decrease by step_size every step_epochs
            if epoch < self.tcp_curriculum_start_epoch:
                self.alpha_di = self.tcp_curriculum_start
            else:
                steps_taken = (
                    epoch - self.tcp_curriculum_start_epoch
                ) // self.tcp_curriculum_step_epochs
                self.alpha_di = max(
                    self.tcp_curriculum_end,
                    self.tcp_curriculum_start
                    - steps_taken * self.tcp_curriculum_step_size,
                )
        else:
            # Linear interpolation mode (original behavior)
            if epoch >= self.tcp_curriculum_epochs:
                self.alpha_di = self.tcp_curriculum_end
            else:
                progress = epoch / self.tcp_curriculum_epochs
                self.alpha_di = self.tcp_curriculum_start + progress * (
                    self.tcp_curriculum_end - self.tcp_curriculum_start
                )

    def _update_feghhi_alpha(self, epoch):
        """
        Update diphone alpha based on Feghhi schedule.

        - First diphone_warmup_epochs: 100% diphone (alpha=0)
        - Then increase phoneme ratio by 0.1 every diphone_ramp_epochs

        Note: In Feghhi schedule, alpha is the PHONEME weight.
        """
        if not self.diphone_feghhi_schedule:
            return

        if epoch < self.diphone_warmup_epochs:
            self._feghhi_alpha = 0.0
        else:
            ramp_step = (
                epoch - self.diphone_warmup_epochs
            ) // self.diphone_ramp_epochs + 1
            self._feghhi_alpha = min(ramp_step * 0.1, self.diphone_max_alpha)

    def _ctc_loss(self, logits, targets, output_lens, target_lens):
        """Compute CTC loss with proper formatting."""
        logits_f32 = logits.float()
        return self.loss_ctc(
            torch.permute(logits_f32.log_softmax(2), [1, 0, 2]),
            targets,
            output_lens,
            target_lens,
        )

    def _compute_tcp_loss(self, model_output, y, y_len, output_lens):
        """
        Compute TCP multi-level loss.

        L_total = α_fused * L_fused + α_mono * L_mono + α_di * L_di + α_context * L_context - λ_gate * H(gates)

        Returns:
            total_loss: Combined loss for backprop
            loss_dict: Dict of individual losses for logging
        """
        loss_dict = {}

        z_fused = model_output["phone_logits"]
        z_mono = model_output["mono_logits"]
        z_diphone_raw = model_output[
            "diphone_logits"
        ]  # [B, T, n_diphones + 1]
        z_context = model_output["context_logits"]
        gates = model_output.get("gates")

        # Primary fused loss
        L_fused = self._ctc_loss(z_fused, y, output_lens, y_len)
        L_fused = torch.sum(L_fused)
        loss_dict["L_fused"] = L_fused.item()

        # Monophone intermediate loss
        L_mono = self._ctc_loss(z_mono, y, output_lens, y_len)
        L_mono = torch.sum(L_mono)
        loss_dict["L_mono"] = L_mono.item()

        # Diphone loss (with diphone targets)
        diphone_y, diphone_len = self._phone_to_diphone_labels(y, y_len)
        L_di = self._ctc_loss(
            z_diphone_raw, diphone_y, output_lens, diphone_len
        )
        L_di = torch.sum(L_di)
        loss_dict["L_di"] = L_di.item()

        # Context head loss
        L_context = self._ctc_loss(z_context, y, output_lens, y_len)
        L_context = torch.sum(L_context)
        loss_dict["L_context"] = L_context.item()

        # Gate entropy regularization (encourage diverse gate usage)
        L_gate = torch.tensor(0.0, device=self.device)
        if gates is not None and self.lambda_gate > 0:
            # Compute entropy: -Σ(g * log(g))
            gate_entropy = -torch.sum(
                gates * torch.log(gates + 1e-8), dim=-1
            ).mean()
            # Maximize entropy (minimize negative entropy)
            L_gate = -self.lambda_gate * gate_entropy
            loss_dict["gate_entropy"] = gate_entropy.item()

        # Update running gate statistics for logging/debugging
        if gates is not None:
            self._update_gate_stats(gates)

        # Total loss
        total_loss = (
            self.alpha_fused * L_fused
            + self.alpha_mono * L_mono
            + self.alpha_di * L_di
            + self.alpha_context * L_context
            + L_gate
        )

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def _compute_standard_loss(self, model_output, y, y_len, output_lens):
        """
        Compute standard CTC loss for non-TCP mode.

        Returns:
            total_loss: CTC loss
            loss_dict: Dict with loss value
        """
        if isinstance(model_output, dict):
            phone_logits = model_output["phone_logits"]
        else:
            phone_logits = model_output

        loss = self._ctc_loss(phone_logits, y, output_lens, y_len)
        loss = torch.sum(loss)

        return loss, {"phone_loss": loss.item(), "total_loss": loss.item()}

    def _train_step(self, X, y, y_len, dayIdx, output_lens):
        """
        Single training step.

        Returns:
            loss: The computed loss value
        """
        if self.use_amp:
            with autocast():
                model_output = self.model.forward(X, dayIdx)

                if self.use_tcp:
                    loss, _ = self._compute_tcp_loss(
                        model_output, y, y_len, output_lens
                    )
                else:
                    loss, _ = self._compute_standard_loss(
                        model_output, y, y_len, output_lens
                    )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            model_output = self.model.forward(X, dayIdx)

            if self.use_tcp:
                loss, _ = self._compute_tcp_loss(
                    model_output, y, y_len, output_lens
                )
            else:
                loss, _ = self._compute_standard_loss(
                    model_output, y, y_len, output_lens
                )

            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

        return loss

    def log_metrics_epoch(
        self, epoch, train_loss, train_per, test_loss, test_per, lr, extra=""
    ):
        """Log metrics with epoch-based format."""
        endTime = time.time()
        elapsed = endTime - self.startTime

        log_msg = (
            f"epoch {epoch}, "
            f"train_loss: {train_loss:>7.4f}, train_per: {train_per:>7.4f}, "
            f"test_loss: {test_loss:>7.4f}, test_per: {test_per:>7.4f}, "
            f"lr: {lr:.6f}, time: {elapsed:>7.1f}s"
        )
        if extra:
            log_msg += f" {extra}"
        print(log_msg)

        try:
            from datetime import datetime

            timestamp = datetime.now().isoformat(timespec="seconds")
            with open(
                os.path.join(self.cfg.outputDir, "training_log.txt"), "a"
            ) as f:
                f.write(f"{timestamp} - {log_msg}\n")
        except Exception as e:
            print(f"Warning: log write failed: {e}")

        self.startTime = time.time()

    def fit(self):
        """
        Epoch-based training loop for TCP Transformer.
        """
        n_epochs = getattr(self.cfg, "nEpochs", 250)

        # Check for new phased LR schedule (takes priority)
        phased_scheduler = create_phased_scheduler(self.cfg, self.optimizer)
        use_phased_lr = phased_scheduler is not None

        if use_phased_lr:
            print(
                f"Using phased LR schedule with {len(phased_scheduler.phases)} phases"
            )

        # Check for new phased loss schedule (takes priority over legacy curriculum)
        loss_defaults = {
            "alpha_fused": self.alpha_fused,
            "alpha_di": self.alpha_di,
            "alpha_mono": self.alpha_mono,
            "alpha_context": self.alpha_context,
            "lambda_gate": self.lambda_gate,
        }
        loss_scheduler = create_loss_scheduler(self.cfg, loss_defaults)
        use_loss_schedule = loss_scheduler is not None

        if use_loss_schedule:
            print(
                f"Using phased loss schedule for: {loss_scheduler.param_names}"
            )

        # Legacy LR decay configuration (only used if no lr_schedule defined)
        # Mode 1: Step decay (gradual) - decrease by lr_step_decay every lr_step_epochs starting at lr_step_start_epoch
        # Mode 2: Instant decay (legacy) - multiply by decay_factor at specific epochs
        lr_step_start_epoch = getattr(self.cfg, "lrStepStartEpoch", None)
        lr_step_epochs = getattr(self.cfg, "lrStepEpochs", 20)
        lr_step_decay = getattr(
            self.cfg, "lrStepDecay", 0.2
        )  # Fraction of initial LR to subtract

        # Legacy instant decay parameters
        decay_epoch = getattr(self.cfg, "lrDecayEpoch", 150)
        decay_factor = getattr(self.cfg, "lrDecayFactor", 0.1)
        decay_epoch_2 = getattr(self.cfg, "lrDecayEpoch2", None)
        decay_factor_2 = getattr(self.cfg, "lrDecayFactor2", 0.1)

        initial_lr = self.optimizer.param_groups[0]["lr"]
        current_lr = initial_lr
        lr_decayed = False
        lr_decayed_2 = False
        last_lr_step = -1  # Track which step we're on for gradual decay

        # Initialize Feghhi alpha if using that schedule
        self._feghhi_alpha = 0.0

        for epoch in range(n_epochs):
            # Reset TCP gate statistics at the start of each epoch
            self._reset_gate_stats()
            # Update loss weights based on epoch
            if use_loss_schedule:
                # Use new phased loss scheduler
                loss_values = loss_scheduler.step(epoch)
                self.alpha_fused = loss_values.get("alpha_fused", self.alpha_fused)
                self.alpha_di = loss_values.get("alpha_di", self.alpha_di)
                self.alpha_mono = loss_values.get(
                    "alpha_mono", self.alpha_mono
                )
                self.alpha_context = loss_values.get(
                    "alpha_context", self.alpha_context
                )
                self.lambda_gate = loss_values.get(
                    "lambda_gate", self.lambda_gate
                )
            else:
                # Legacy: use old curriculum methods
                self._update_tcp_alpha_di(epoch)
            self._update_feghhi_alpha(epoch)

            # LR scheduling
            if use_phased_lr:
                # Use new phased scheduler
                current_lr = phased_scheduler.step(epoch)
            elif epoch < self.warmup_epochs:
                # Legacy: LR warmup
                warmup_lr = initial_lr * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr
                current_lr = warmup_lr
            elif epoch == self.warmup_epochs:
                # Legacy: End of warmup - set to initial LR
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = initial_lr
                current_lr = initial_lr
            # Legacy: Gradual step decay mode
            elif (
                lr_step_start_epoch is not None
                and epoch >= lr_step_start_epoch
            ):
                current_step = (epoch - lr_step_start_epoch) // lr_step_epochs
                if current_step > last_lr_step:
                    # New step reached - decrease LR by lr_step_decay * initial_lr
                    new_lr = max(
                        0.00001,
                        initial_lr
                        * (1.0 - (current_step + 1) * lr_step_decay),
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr
                    current_lr = new_lr
                    last_lr_step = current_step
            # Legacy: instant decay mode
            elif lr_step_start_epoch is None:
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

                # Apply augmentations
                X_aug = self.augmenter(X, X_44)

                # Training step
                loss = self._train_step(X_aug, y, y_len, dayIdx, output_lens)
                epoch_losses.append(loss.item())

            # Evaluation at end of each epoch
            with torch.no_grad():
                self.model.eval()
                self.augmenter.eval()

                train_loss, train_per = self._evaluate_loader(
                    self.train_loader, max_batches=10
                )
                test_loss, test_per = self._evaluate_loader(self.test_loader)

            # Build extra info string
            extra = ""
            if self.use_tcp:
                if use_loss_schedule:
                    # Log all scheduled loss weights
                    extra += f"α_f={self.alpha_fused:.2f} α_m={self.alpha_mono:.2f} α_d={self.alpha_di:.2f} α_c={self.alpha_context:.2f} λ_g={self.lambda_gate:.3f}"
                else:
                    extra += f"α_di={self.alpha_di:.2f}"

                # If using learned gating, append average gate values
                gate_stats = self._get_gate_stats()
                if gate_stats is not None:
                    g_mono, g_di, g_context = gate_stats
                    extra += f" gates=[m={g_mono:.2f}, d={g_di:.2f}, c={g_context:.2f}]"
            if self.diphone_feghhi_schedule:
                extra += f" feghhi_α={self._feghhi_alpha:.2f}"

            # DietCORP evaluation (periodic, slow)
            # NOTE: DietCORP needs gradients for adaptation, so no torch.no_grad() here
            if (
                self.dietcorp_eval_frequency > 0
                and epoch > 0
                and epoch % self.dietcorp_eval_frequency == 0
            ):
                conf_thresh = getattr(
                    self, "dietcorp_confidence_threshold", 0.5
                )
                print(
                    f"  Running DietCORP evaluation (views={self.dietcorp_views}, lr={self.dietcorp_lr}, conf_thresh={conf_thresh})..."
                )
                # Reset debug flags and stats for each evaluation
                if hasattr(self, "_dietcorp_debug_printed"):
                    delattr(self, "_dietcorp_debug_printed")
                if hasattr(self, "_dietcorp_weight_debug_printed"):
                    delattr(self, "_dietcorp_weight_debug_printed")
                if hasattr(self, "_dietcorp_confidence_stats"):
                    delattr(self, "_dietcorp_confidence_stats")

                # Save patch_embed state BEFORE DietCORP evaluation
                orig_patch_state = copy.deepcopy(
                    self.model.patch_embed.state_dict()
                )
                orig_weight = self.model.patch_embed.weight.data.clone()

                self.model.eval()
                _, test_per_dietcorp = self._evaluate_loader(
                    self.test_loader, use_dietcorp=True
                )

                # Show cumulative weight change across entire test set
                final_weight = self.model.patch_embed.weight.data
                cumulative_change = (
                    (final_weight - orig_weight).abs().mean().item()
                )
                print(
                    f"  [DietCORP] Cumulative weight change over test set: {cumulative_change:.6f}"
                )

                # Show confidence filtering statistics
                if hasattr(self, "_dietcorp_confidence_stats"):
                    stats = self._dietcorp_confidence_stats
                    total = stats["adapted"] + stats["skipped"]
                    avg_conf = stats["total_conf"] / total if total > 0 else 0
                    print(
                        f"  [DietCORP] Adapted: {stats['adapted']}/{total} trials ({100*stats['adapted']/total:.1f}%), avg_conf={avg_conf:.3f}"
                    )

                # Restore patch_embed state AFTER DietCORP evaluation
                self.model.patch_embed.load_state_dict(orig_patch_state)

                extra += f" dietcorp_per={test_per_dietcorp:.4f}"
                print(
                    f"  DietCORP test PER: {test_per_dietcorp:.4f} (vs baseline: {test_per:.4f})"
                )

            # Log and save
            self.log_metrics_epoch(
                epoch,
                train_loss,
                train_per,
                test_loss,
                test_per,
                current_lr,
                extra,
            )
            self.save_checkpoint(self.model, test_per)
            self.save_stats(train_loss, train_per, test_loss, test_per)

    def _evaluate_loader(self, loader, max_batches=None, use_dietcorp=False):
        """
        Evaluate model on a data loader, returning avg loss and PER.

        Uses only the fused output (phone_logits) for evaluation.

        Args:
            loader: DataLoader to evaluate
            max_batches: Optional max number of batches to evaluate
            use_dietcorp: If True, apply DietCORP adaptation per-trial (slow)
        """
        all_loss = []
        total_edit_distance = 0
        total_seq_length = 0

        for batch_idx, batch_data in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            X, X_44, y, X_len, y_len, dayIdx = self._unpack_batch(batch_data)
            output_lens = self._compute_output_lens(X_len)

            if use_dietcorp:
                # DietCORP: adapt per-trial (very slow)
                all_preds = []
                for i in range(X.shape[0]):
                    X_single = X[i : i + 1]
                    dayIdx_single = dayIdx[i : i + 1]
                    # continuous=True means weights persist across trials
                    model_output = self._adapt_and_infer_dietcorp(
                        X_single, dayIdx_single, continuous=True
                    )

                    if isinstance(model_output, dict):
                        pred_single = model_output["phone_logits"]
                    else:
                        pred_single = model_output
                    all_preds.append(pred_single)

                pred = torch.cat(all_preds, dim=0)
            else:
                # Standard forward pass
                if self.use_amp:
                    with autocast():
                        model_output = self.model.forward(X, dayIdx)
                else:
                    model_output = self.model.forward(X, dayIdx)

                # Get phone logits (fused output for TCP)
                if isinstance(model_output, dict):
                    pred = model_output["phone_logits"]
                else:
                    pred = model_output

            pred_f32 = pred.float()

            # Compute loss
            loss = self.loss_ctc(
                torch.permute(pred_f32.log_softmax(2), [1, 0, 2]),
                y,
                output_lens,
                y_len,
            )
            loss = torch.sum(loss)
            all_loss.append(loss.cpu().detach().numpy())

            # Compute PER via greedy decoding
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
