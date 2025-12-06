"""
Epoch-based CTC trainer for Transformer models.

Key features:
- Epoch-based training with configurable number of epochs
- Flexible phased LR schedules (warmup, hold, cosine, step, exponential)
- Gradient clipping for stability
- Diphone auxiliary loss with alpha ramping
- Intermediate CTC loss from middle layer
- Mixed precision (FP16) training support
- CR-CTC consistency regularization (optional)
- DietCORP test-time adaptation (optional)
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
from neural_decoder.schedulers import create_phased_scheduler


class TransformerCTCTrainer(BaseTrainer):
    """
    CTC Trainer designed for Transformer models with epoch-based training.
    
    Supports:
    - Step decay LR schedule
    - Diphone auxiliary loss with ramping α
    - Intermediate CTC loss with weight λ
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
        
        # Part 2 features
        self.use_diphone = getattr(model, "use_diphone", False)
        self.use_intermediate_ctc = getattr(model, "use_intermediate_ctc", False)
        
        # Diphone loss settings
        # Feghhi schedule: Start with 100% diphone, ramp UP phoneme ratio to max_alpha
        # Our original: Start with 100% phoneme, ramp UP diphone ratio to max_alpha
        self.diphone_feghhi_schedule = getattr(cfg, "diphone_feghhi_schedule", False)
        self.diphone_alpha = 0.0  # Current alpha (ramped during training)
        self.diphone_max_alpha = getattr(cfg, "diphone_max_alpha", 0.6)
        self.diphone_ramp_epochs = getattr(cfg, "diphone_ramp_epochs", 25)
        self.diphone_warmup_epochs = getattr(cfg, "diphone_warmup_epochs", 10)  # For Feghhi: pure diphone epochs
        
        # Intermediate CTC loss weight
        self.intermediate_ctc_weight = getattr(cfg, "intermediate_ctc_weight", 0.3)
        
        # Mixed precision
        self.use_amp = getattr(cfg, "useMixedPrecision", False)
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Load diphone vocab if using diphone loss
        self.diphone_vocab = None
        if self.use_diphone:
            diphone_vocab_path = getattr(cfg, "diphone_vocab_path", "data/diphone_vocab.pkl")
            # Resolve relative path from original working directory (Hydra changes cwd)
            if not os.path.isabs(diphone_vocab_path):
                try:
                    import hydra
                    orig_cwd = hydra.utils.get_original_cwd()
                    diphone_vocab_path = os.path.join(orig_cwd, diphone_vocab_path)
                except Exception:
                    pass  # Fall back to relative path if hydra not available
            
            if os.path.exists(diphone_vocab_path):
                with open(diphone_vocab_path, "rb") as f:
                    self.diphone_vocab = pickle.load(f)
                print(f"Loaded diphone vocab: {len(self.diphone_vocab['diphone_to_idx'])} diphones")
            else:
                print(f"WARNING: Diphone vocab not found at {diphone_vocab_path}")
        
        # Part 3: LR warmup
        self.warmup_epochs = getattr(cfg, "lrWarmupEpochs", 5)
        
        # Part 3: CR-CTC consistency regularization
        self.use_crctc = getattr(cfg, "useCRCTC", False)
        # Max consistency weight (β); may be ramped during training
        self.crctc_beta_max = getattr(cfg, "crctcBeta", 0.2)
        self.crctc_ramp_epochs = getattr(cfg, "crctcRampEpochs", 0)
        self.crctc_start_epoch = getattr(cfg, "crctcStartEpoch", 0)
        self.crctc_use_gradient_accum = getattr(cfg, "crctcGradientAccum", True)
        self.crctc_beta_current = 0.0  # actual β used at current epoch
        # Heavy view masking parameters (configurable)
        self.crctc_heavy_masks = getattr(cfg, "crctcHeavyMasks", 50)
        self.crctc_heavy_fraction = getattr(cfg, "crctcHeavyFraction", 0.375)
        
        # Part 3: DietCORP test-time adaptation
        self.use_dietcorp = getattr(cfg, "useDietCORP", False)
        self.dietcorp_lr = getattr(cfg, "dietcorpLR", 0.0001)
        self.dietcorp_views = getattr(cfg, "dietcorpViews", 64)
        self.dietcorp_eval_frequency = getattr(cfg, "dietcorpEvalFrequency", 0)
        self.dietcorp_confidence_threshold = getattr(cfg, "dietcorpConfidenceThreshold", 0.5)
    
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
        return ((X_len - self.model.kernelLen) / self.model.strideLen).to(torch.int32)
    
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
        
        # Diphone sequences are 1 shorter than phone sequences
        diphone_y = torch.zeros(B, max_len - 1, dtype=torch.int32, device=y.device)
        diphone_len = torch.zeros(B, dtype=torch.int32, device=y.device)
        
        for b in range(B):
            seq_len = y_len[b].item()
            if seq_len < 2:
                continue
            
            for i in range(seq_len - 1):
                p1 = y[b, i].item()
                p2 = y[b, i + 1].item()
                diphone = (p1, p2)
                # Use 0 (blank) for unknown diphones
                diphone_idx = diphone_to_idx.get(diphone, 0)
                diphone_y[b, i] = diphone_idx
            
            diphone_len[b] = seq_len - 1
        
        return diphone_y, diphone_len
    
    def _update_diphone_alpha(self, epoch):
        """
        Update diphone loss weight based on epoch.
        
        Two modes:
        1. Feghhi schedule (diphone_feghhi_schedule=True):
           - First diphone_warmup_epochs: 100% diphone (alpha=0 means phoneme weight=0)
           - Then increase phoneme ratio by 0.1 every diphone_ramp_epochs until max_alpha
           - Loss = alpha * phone + (1-alpha) * diphone
           
        2. Original schedule (diphone_feghhi_schedule=False):
           - Start with 100% phoneme
           - Increase diphone ratio by 0.1 every diphone_ramp_epochs until max_alpha
           - Loss = (1-alpha) * phone + alpha * diphone
        """
        if not self.use_diphone:
            return
        
        if self.diphone_feghhi_schedule:
            # Feghhi: alpha is PHONEME weight, starts at 0
            if epoch < self.diphone_warmup_epochs:
                # Pure diphone training
                self.diphone_alpha = 0.0
            else:
                # Ramp up phoneme weight by 0.1 every ramp_epochs
                ramp_step = (epoch - self.diphone_warmup_epochs) // self.diphone_ramp_epochs + 1
                self.diphone_alpha = min(ramp_step * 0.1, self.diphone_max_alpha)
        else:
            # Original: alpha is DIPHONE weight, starts at 0
            ramp_step = epoch // self.diphone_ramp_epochs
            self.diphone_alpha = min(ramp_step * 0.1, self.diphone_max_alpha)
    
    def _update_crctc_beta(self, epoch):
        """
        Update CR-CTC consistency weight β based on epoch.
        
        Supports optional warm-start and linear ramp:
          - Before crctc_start_epoch: β = 0
          - During ramp: β increases linearly from 0 → crctc_beta_max
          - After ramp: β = crctc_beta_max
        """
        if not self.use_crctc:
            self.crctc_beta_current = 0.0
            return
        
        # No ramp: use full beta from the beginning
        if self.crctc_ramp_epochs <= 0 and self.crctc_start_epoch <= 0:
            self.crctc_beta_current = self.crctc_beta_max
            return
        
        if epoch < self.crctc_start_epoch:
            self.crctc_beta_current = 0.0
            return
        
        if self.crctc_ramp_epochs <= 0:
            # Start epoch specified but no ramp length: jump to full beta
            self.crctc_beta_current = self.crctc_beta_max
            return
        
        # Linear ramp
        progress = (epoch - self.crctc_start_epoch) / float(self.crctc_ramp_epochs)
        progress = max(0.0, min(1.0, progress))
        self.crctc_beta_current = self.crctc_beta_max * progress
    
    def _kl_consistency(self, logits1, logits2):
        """
        Compute symmetric KL divergence with stop-gradient for CR-CTC.
        
        Args:
            logits1: [B, T, C] logits from view 1
            logits2: [B, T, C] logits from view 2
            
        Returns:
            Symmetric KL divergence loss
        """
        # Get probabilities
        p1 = F.softmax(logits1, dim=-1)
        p2 = F.softmax(logits2, dim=-1)
        
        # Log probabilities for KL computation
        log_p1 = F.log_softmax(logits1, dim=-1)
        log_p2 = F.log_softmax(logits2, dim=-1)
        
        # Symmetric KL with stop-gradient on targets
        # KL(P1 || stopgrad(P2)) + KL(P2 || stopgrad(P1))
        kl1 = F.kl_div(log_p1, p2.detach(), reduction='batchmean')
        kl2 = F.kl_div(log_p2, p1.detach(), reduction='batchmean')
        
        return 0.5 * (kl1 + kl2)
    
    def _apply_time_masking(self, x, num_masks=20, max_mask_fraction=0.075):
        """
        Apply time-masking augmentation to input tensor.
        
        Used for CR-CTC heavy view and DietCORP adaptation views.
        Note: This applies masking to raw input (before patch embedding),
        so we use zeros rather than the model's learnable mask token
        (which is in hidden_dim space, not input space).
        
        Args:
            x: [B, T, C] input tensor (C = neural_dim, e.g. 256)
            num_masks: Number of masks to apply
            max_mask_fraction: Maximum fraction of sequence per mask
            
        Returns:
            x_masked: [B, T, C] masked tensor
        """
        B, T, C = x.shape
        x_masked = x.clone()
        
        for _ in range(num_masks):
            # Random mask length
            mask_len = int(T * max_mask_fraction * torch.rand(1).item())
            if mask_len == 0:
                continue
            
            # Random start position for each batch item
            for b in range(B):
                start = torch.randint(0, max(1, T - mask_len), (1,)).item()
                end = min(start + mask_len, T)
                # Zero out the masked region (in input space)
                x_masked[b, start:end, :] = 0.0
        
        return x_masked
    
    def _adapt_and_infer_dietcorp(self, X, dayIdx, continuous=True):
        """
        Single-trial DietCORP adaptation and inference (Feghhi et al. 2025).
        
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
                       If False, reset after each trial (for fair single-trial comparison).
            
        Returns:
            final_output: Model output after adaptation
        """
        # Confidence threshold for adaptation (skip noisy pseudo-labels)
        confidence_threshold = getattr(self, 'dietcorp_confidence_threshold', 0.5)
        
        # Save original patch embedding state (only needed if not continuous)
        if not continuous:
            orig_state = copy.deepcopy(self.model.patch_embed.state_dict())
        
        # Step 1: Get pseudo-label by greedy decoding (eval mode, no augmentation)
        self.model.eval()
        with torch.no_grad():
            unadapted_out = self.model.forward(X, dayIdx)
            if isinstance(unadapted_out, dict):
                pseudo_logits = unadapted_out['phone_logits']  # [1, T, C]
            else:
                pseudo_logits = unadapted_out
            
            # Compute confidence: max softmax probability at each time step
            pseudo_probs = F.softmax(pseudo_logits, dim=-1)  # [1, T, C]
            max_probs, _ = pseudo_probs.max(dim=-1)  # [1, T]
            avg_confidence = max_probs.mean().item()
            
            # Track confidence statistics
            if not hasattr(self, '_dietcorp_confidence_stats'):
                self._dietcorp_confidence_stats = {'adapted': 0, 'skipped': 0, 'total_conf': 0.0}
            self._dietcorp_confidence_stats['total_conf'] += avg_confidence
            
            # Skip adaptation for low-confidence predictions
            if avg_confidence < confidence_threshold:
                self._dietcorp_confidence_stats['skipped'] += 1
                # Return unadapted output directly
                for param in self.model.parameters():
                    param.requires_grad = True
                return unadapted_out
            
            self._dietcorp_confidence_stats['adapted'] += 1
            
            # Greedy decode: get the most likely phoneme at each time step
            raw_sequence = pseudo_logits.argmax(dim=-1).squeeze(0)  # [T]
            
            # CTC collapse: remove blanks (class 0) and consecutive duplicates
            # This is the standard CTC decoding procedure
            collapsed = []
            prev = -1
            for idx in raw_sequence.tolist():
                if idx != 0 and idx != prev:  # Skip blanks and duplicates
                    collapsed.append(idx)
                prev = idx
            
            # Handle edge case: if all blanks, use a dummy target
            if len(collapsed) == 0:
                collapsed = [1]  # Use first non-blank class
            
            pseudo_sequence = torch.tensor(collapsed, dtype=torch.long, device=X.device)
            pseudo_length = len(collapsed)
        
        # Step 2: Create Z augmented views using SAME augmentations as training
        # Paper: "white noise, baseline shift, and time-masking to each copy"
        views = []
        self.augmenter.train()  # Enable white noise + baseline shift
        for _ in range(self.dietcorp_views):
            aug_X = self.augmenter(X.clone(), None)
            views.append(aug_X)
        
        # Stack views: [Z, T, C]
        views_batch = torch.cat(views, dim=0)
        dayIdx_expanded = dayIdx.expand(self.dietcorp_views)
        
        # Freeze all parameters except patch embedding
        for name, param in self.model.named_parameters():
            param.requires_grad = 'patch_embed' in name
        
        # Put model in TRAIN mode so it applies time-masking with learnable mask token
        self.model.train()
        
        # Forward pass on augmented views (model applies time-masking internally)
        if self.use_amp:
            with autocast():
                adapted_out = self.model.forward(views_batch, dayIdx_expanded)
        else:
            adapted_out = self.model.forward(views_batch, dayIdx_expanded)
        
        if isinstance(adapted_out, dict):
            adapted_logits = adapted_out['phone_logits']  # [Z, T, num_classes]
        else:
            adapted_logits = adapted_out
        
        # Step 3: Compute CTC loss between augmented predictions and pseudo-label
        # Repeat pseudo-label for each view (CTC expects flat targets)
        pseudo_targets = pseudo_sequence.unsqueeze(0).expand(self.dietcorp_views, -1)  # [Z, L]
        
        # CTC loss setup
        log_probs = F.log_softmax(adapted_logits.float(), dim=-1)  # [Z, T, C]
        log_probs = log_probs.permute(1, 0, 2)  # [T, Z, C] for CTC
        
        T = log_probs.shape[0]
        input_lengths = torch.full((self.dietcorp_views,), T, dtype=torch.long, device=X.device)
        target_lengths = torch.full((self.dietcorp_views,), pseudo_length, dtype=torch.long, device=X.device)
        
        # CTC loss with collapsed pseudo-labels
        adapt_loss = F.ctc_loss(
            log_probs, pseudo_targets,
            input_lengths, target_lengths,
            blank=0, reduction='mean', zero_infinity=True
        )
        
        # Step 4: Single gradient step with clipping
        adapt_loss.backward()
        
        # Gradient clipping to 0.5 (as per paper)
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in self.model.named_parameters() if 'patch_embed' in n],
            max_norm=0.5
        )
        
        # DEBUG: Print on first trial only
        if not hasattr(self, '_dietcorp_debug_printed'):
            patch_params = [(n, p) for n, p in self.model.named_parameters() if 'patch_embed' in n]
            print(f"  [DietCORP] CTC Loss: {adapt_loss.item():.4f}")
            for n, p in patch_params:
                if p.grad is not None:
                    print(f"  [DietCORP] {n}: grad_norm={p.grad.norm().item():.4f} (after clip)")
            self._dietcorp_debug_printed = True
        
        # Apply gradient update (paper LR = 5e-4)
        # DEBUG: Check weight change
        weight_before = self.model.patch_embed.weight.data.clone()
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'patch_embed' in name and param.grad is not None:
                    # Must use .data.sub_() for in-place modification!
                    # param -= x doesn't modify the actual model parameter
                    param.data.sub_(self.dietcorp_lr * param.grad)
        
        # DEBUG: Measure weight change
        weight_after = self.model.patch_embed.weight.data
        weight_diff_mean = (weight_after - weight_before).abs().mean().item()
        weight_diff_max = (weight_after - weight_before).abs().max().item()
        if not hasattr(self, '_dietcorp_weight_debug_printed'):
            print(f"  [DietCORP] Weight change: mean={weight_diff_mean:.6f}, max={weight_diff_max:.6f}")
            self._dietcorp_weight_debug_printed = True
        
        # Clear gradients
        self.model.zero_grad()
        
        # DEBUG: Compare predictions before vs after
        # Get unadapted prediction for comparison
        self.model.eval()
        self.augmenter.eval()
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    final_out = self.model.forward(X, dayIdx)
            else:
                final_out = self.model.forward(X, dayIdx)
            
            # Check if predictions actually changed
            if not hasattr(self, '_dietcorp_pred_debug_printed'):
                if isinstance(final_out, dict):
                    final_logits = final_out['phone_logits']
                else:
                    final_logits = final_out
                # Compare with pseudo_logits (unadapted)
                pred_diff = (final_logits - pseudo_logits).abs().mean().item()
                print(f"  [DietCORP] Prediction change: {pred_diff:.6f}")
                self._dietcorp_pred_debug_printed = True
        
        # Reset weights only if not using continuous adaptation
        if not continuous:
            # For periodic evaluation during training with fair single-trial comparison
            self.model.patch_embed.load_state_dict(orig_state)
        
        # Re-enable gradients on all parameters for subsequent training
        for param in self.model.parameters():
            param.requires_grad = True
        
        return final_out
    
    def _train_step_standard(self, X, y, y_len, dayIdx, output_lens):
        """
        Standard training step without CR-CTC.
        
        Returns:
            loss: The computed loss value
        """
        if self.use_amp:
            with autocast():
                model_output = self.model.forward(X, dayIdx)
                loss, _ = self._compute_loss(model_output, y, y_len, output_lens)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            model_output = self.model.forward(X, dayIdx)
            loss, _ = self._compute_loss(model_output, y, y_len, output_lens)
            
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            self.optimizer.step()
        
        return loss
    
    def _train_step_crctc(self, X, X_44, y, y_len, dayIdx, output_lens):
        """
        CR-CTC training step with two augmented views.
        
        View 1: Standard augmentation + standard time-masking
        View 2: Standard augmentation + heavy time-masking (2.5x more masks)
        
        Returns:
            loss: Combined CTC + consistency loss
        """
        # View 1: standard augmentation (time-masking happens in model)
        X_v1 = self.augmenter(X.clone(), X_44)
        
        # View 2: standard augmentation + extra heavy time-masking
        X_v2 = self.augmenter(X.clone(), X_44)
        # Apply additional heavy time-masking for view 2 (before model does its own)
        X_v2 = self._apply_time_masking(
            X_v2, 
            num_masks=self.crctc_heavy_masks, 
            max_mask_fraction=self.crctc_heavy_fraction
        )
        
        if self.crctc_use_gradient_accum:
            # Process views sequentially with gradient accumulation
            total_loss = torch.tensor(0.0, device=self.device)
            
            # View 1
            if self.use_amp:
                with autocast():
                    out1 = self.model.forward(X_v1, dayIdx)
                    loss1, _ = self._compute_loss(out1, y, y_len, output_lens)
                self.scaler.scale(loss1).backward(retain_graph=True)
            else:
                out1 = self.model.forward(X_v1, dayIdx)
                loss1, _ = self._compute_loss(out1, y, y_len, output_lens)
                loss1.backward(retain_graph=True)
            
            total_loss = total_loss + loss1.detach()
            
            # View 2
            if self.use_amp:
                with autocast():
                    out2 = self.model.forward(X_v2, dayIdx)
                    loss2, _ = self._compute_loss(out2, y, y_len, output_lens)
                self.scaler.scale(loss2).backward(retain_graph=True)
            else:
                out2 = self.model.forward(X_v2, dayIdx)
                loss2, _ = self._compute_loss(out2, y, y_len, output_lens)
                loss2.backward(retain_graph=True)
            
            total_loss = total_loss + loss2.detach()
            
            # Consistency loss
            if isinstance(out1, dict):
                logits1 = out1['phone_logits']
                logits2 = out2['phone_logits']
            else:
                logits1 = out1
                logits2 = out2
            
            if self.use_amp:
                with autocast():
                    consistency_loss = self._kl_consistency(logits1, logits2)
                self.scaler.scale(self.crctc_beta_current * consistency_loss).backward()
            else:
                consistency_loss = self._kl_consistency(logits1, logits2)
                (self.crctc_beta_current * consistency_loss).backward()
            
            total_loss = total_loss + self.crctc_beta_current * consistency_loss.detach()
            
            # Step optimizer
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                self.optimizer.step()
            
            return total_loss
        else:
            # Process both views in parallel (requires more memory)
            if self.use_amp:
                with autocast():
                    out1 = self.model.forward(X_v1, dayIdx)
                    out2 = self.model.forward(X_v2, dayIdx)
                    
                    loss1, _ = self._compute_loss(out1, y, y_len, output_lens)
                    loss2, _ = self._compute_loss(out2, y, y_len, output_lens)
                    
                    if isinstance(out1, dict):
                        logits1 = out1['phone_logits']
                        logits2 = out2['phone_logits']
                    else:
                        logits1 = out1
                        logits2 = out2
                    
                    consistency_loss = self._kl_consistency(logits1, logits2)
                    total_loss = loss1 + loss2 + self.crctc_beta_current * consistency_loss
                
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out1 = self.model.forward(X_v1, dayIdx)
                out2 = self.model.forward(X_v2, dayIdx)
                
                loss1, _ = self._compute_loss(out1, y, y_len, output_lens)
                loss2, _ = self._compute_loss(out2, y, y_len, output_lens)
                
                if isinstance(out1, dict):
                    logits1 = out1['phone_logits']
                    logits2 = out2['phone_logits']
                else:
                    logits1 = out1
                    logits2 = out2
                
                consistency_loss = self._kl_consistency(logits1, logits2)
                total_loss = loss1 + loss2 + self.crctc_beta_current * consistency_loss
                
                total_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                self.optimizer.step()
            
            return total_loss

    def _compute_loss(self, model_output, y, y_len, output_lens):
        """
        Compute combined loss with optional diphone and intermediate CTC.
        
        Returns:
            total_loss: Combined loss for backprop
            loss_dict: Dict of individual losses for logging
        """
        loss_dict = {}
        
        # Handle both simple output and dict output
        if isinstance(model_output, dict):
            phone_logits = model_output['phone_logits']
            diphone_logits = model_output.get('diphone_logits')
            intermediate_logits = model_output.get('intermediate_logits')
        else:
            phone_logits = model_output
            diphone_logits = None
            intermediate_logits = None
        
        # CTC loss requires float32 (doesn't support FP16)
        phone_logits = phone_logits.float()
        if diphone_logits is not None:
            diphone_logits = diphone_logits.float()
        if intermediate_logits is not None:
            intermediate_logits = intermediate_logits.float()
        
        # Primary phoneme CTC loss
        phone_loss = self.loss_ctc(
            torch.permute(phone_logits.log_softmax(2), [1, 0, 2]),
            y,
            output_lens,
            y_len,
        )
        phone_loss = torch.sum(phone_loss)
        loss_dict['phone_loss'] = phone_loss.item()
        
        # Compute diphone loss if enabled
        diphone_loss = None
        if self.use_diphone and diphone_logits is not None:
            diphone_y, diphone_len = self._phone_to_diphone_labels(y, y_len)
            
            # Diphone output length is same as phone output length
            diphone_loss = self.loss_ctc(
                torch.permute(diphone_logits.log_softmax(2), [1, 0, 2]),
                diphone_y,
                output_lens,
                diphone_len,
            )
            diphone_loss = torch.sum(diphone_loss)
            loss_dict['diphone_loss'] = diphone_loss.item()
        
        # Combine losses based on schedule
        if self.use_diphone and diphone_loss is not None:
            if self.diphone_feghhi_schedule:
                # Feghhi: alpha is phoneme weight
                # Loss = alpha * phone + (1-alpha) * diphone
                total_loss = self.diphone_alpha * phone_loss + (1 - self.diphone_alpha) * diphone_loss
            else:
                # Original: alpha is diphone weight
                # Loss = (1-alpha) * phone + alpha * diphone
                total_loss = (1 - self.diphone_alpha) * phone_loss + self.diphone_alpha * diphone_loss
        else:
            total_loss = phone_loss
        
        # Intermediate CTC loss (if enabled)
        if self.use_intermediate_ctc and intermediate_logits is not None:
            inter_loss = self.loss_ctc(
                torch.permute(intermediate_logits.log_softmax(2), [1, 0, 2]),
                y,
                output_lens,
                y_len,
            )
            inter_loss = torch.sum(inter_loss)
            loss_dict['intermediate_loss'] = inter_loss.item()
            
            total_loss = total_loss + self.intermediate_ctc_weight * inter_loss
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
    
    def log_metrics_epoch(self, epoch, train_loss, train_per, test_loss, test_per, lr, extra=""):
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
        Epoch-based training loop with Part 2 and Part 3 features.
        """
        n_epochs = getattr(self.cfg, "nEpochs", 250)
        
        # Check for new phased LR schedule (takes priority)
        phased_scheduler = create_phased_scheduler(self.cfg, self.optimizer)
        use_phased_lr = phased_scheduler is not None
        
        if use_phased_lr:
            print(f"Using phased LR schedule with {len(phased_scheduler.phases)} phases")
        
        # Legacy LR decay config (only used if no lr_schedule)
        decay_epoch = getattr(self.cfg, "lrDecayEpoch", 150)
        decay_factor = getattr(self.cfg, "lrDecayFactor", 0.1)
        
        # Support for second LR decay
        decay_epoch_2 = getattr(self.cfg, "lrDecayEpoch2", None)
        decay_factor_2 = getattr(self.cfg, "lrDecayFactor2", 0.1)
        
        initial_lr = self.optimizer.param_groups[0]["lr"]
        current_lr = initial_lr
        lr_decayed = False
        lr_decayed_2 = False
        
        for epoch in range(n_epochs):
            # Update auxiliary loss weights based on epoch
            self._update_diphone_alpha(epoch)
            self._update_crctc_beta(epoch)
            
            # LR scheduling
            if use_phased_lr:
                # Use new phased scheduler
                current_lr = phased_scheduler.step(epoch)
            elif epoch < self.warmup_epochs:
                # Legacy: LR warmup - linear increase from 0 to initial_lr
                warmup_lr = initial_lr * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr
                current_lr = warmup_lr
            # Legacy: Apply first LR decay at specified epoch
            elif epoch == decay_epoch and not lr_decayed:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= decay_factor
                current_lr = self.optimizer.param_groups[0]["lr"]
                lr_decayed = True
            # Legacy: Apply second LR decay if specified
            elif decay_epoch_2 is not None and epoch == decay_epoch_2 and not lr_decayed_2:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= decay_factor_2
                current_lr = self.optimizer.param_groups[0]["lr"]
                lr_decayed_2 = True
            elif epoch == self.warmup_epochs and not lr_decayed:
                # Legacy: Ensure we're at full initial_lr after warmup
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = initial_lr
                current_lr = initial_lr
            
            # Training epoch
            self.model.train()
            self.augmenter.train()
            
            epoch_losses = []
            for batch_data in self.train_loader:
                X, X_44, y, X_len, y_len, dayIdx = self._unpack_batch(batch_data)
                
                self.optimizer.zero_grad()
                output_lens = self._compute_output_lens(X_len)
                
                if self.use_crctc:
                    # CR-CTC: two views with different augmentations
                    loss = self._train_step_crctc(X, X_44, y, y_len, dayIdx, output_lens)
                else:
                    # Standard training
                    X_aug = self.augmenter(X, X_44)
                    loss = self._train_step_standard(X_aug, y, y_len, dayIdx, output_lens)
                
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
            if self.use_diphone:
                extra += f"α={self.diphone_alpha:.2f}"
            if self.use_crctc and self.crctc_beta_current > 0:
                extra += f" β={self.crctc_beta_current:.2f}"
            
            # DietCORP evaluation (periodic, slow)
            # NOTE: DietCORP needs gradients for adaptation, so no torch.no_grad() here
            if (self.dietcorp_eval_frequency > 0 and 
                epoch > 0 and 
                epoch % self.dietcorp_eval_frequency == 0):
                conf_thresh = getattr(self, 'dietcorp_confidence_threshold', 0.5)
                print(f"  Running DietCORP evaluation (views={self.dietcorp_views}, lr={self.dietcorp_lr}, conf_thresh={conf_thresh})...")
                # Reset debug flags and stats for each evaluation
                if hasattr(self, '_dietcorp_debug_printed'):
                    delattr(self, '_dietcorp_debug_printed')
                if hasattr(self, '_dietcorp_weight_debug_printed'):
                    delattr(self, '_dietcorp_weight_debug_printed')
                if hasattr(self, '_dietcorp_pred_debug_printed'):
                    delattr(self, '_dietcorp_pred_debug_printed')
                if hasattr(self, '_dietcorp_confidence_stats'):
                    delattr(self, '_dietcorp_confidence_stats')
                
                # Save patch_embed state BEFORE DietCORP evaluation
                # (continuous adaptation happens across test set, then we restore)
                orig_patch_state = copy.deepcopy(self.model.patch_embed.state_dict())
                orig_weight = self.model.patch_embed.weight.data.clone()
                
                self.model.eval()
                _, test_per_dietcorp = self._evaluate_loader(
                    self.test_loader, use_dietcorp=True
                )
                
                # Show cumulative weight change across entire test set
                final_weight = self.model.patch_embed.weight.data
                cumulative_change = (final_weight - orig_weight).abs().mean().item()
                print(f"  [DietCORP] Cumulative weight change over test set: {cumulative_change:.6f}")
                
                # Show confidence filtering statistics
                if hasattr(self, '_dietcorp_confidence_stats'):
                    stats = self._dietcorp_confidence_stats
                    total = stats['adapted'] + stats['skipped']
                    avg_conf = stats['total_conf'] / total if total > 0 else 0
                    print(f"  [DietCORP] Adapted: {stats['adapted']}/{total} trials ({100*stats['adapted']/total:.1f}%), avg_conf={avg_conf:.3f}")
                
                # Restore patch_embed state AFTER DietCORP evaluation
                # (so training continues with original weights)
                self.model.patch_embed.load_state_dict(orig_patch_state)
                
                extra += f" dietcorp_per={test_per_dietcorp:.4f}"
                print(f"  DietCORP test PER: {test_per_dietcorp:.4f} (vs baseline: {test_per:.4f})")
            
            # Log and save
            self.log_metrics_epoch(epoch, train_loss, train_per, test_loss, test_per, current_lr, extra)
            self.save_checkpoint(self.model, test_per)
            self.save_stats(train_loss, train_per, test_loss, test_per)
    
    def _evaluate_loader(self, loader, max_batches=None, use_dietcorp=False):
        """
        Evaluate model on a data loader, returning avg loss and PER.
        
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
                # DietCORP: adapt per-trial with continuous calibration
                # (adaptation accumulates across trials, matching paper)
                all_preds = []
                for i in range(X.shape[0]):
                    X_single = X[i:i+1]
                    dayIdx_single = dayIdx[i:i+1]
                    # continuous=True means weights persist across trials
                    model_output = self._adapt_and_infer_dietcorp(X_single, dayIdx_single, continuous=True)
                    
                    if isinstance(model_output, dict):
                        pred = model_output['phone_logits']
                    else:
                        pred = model_output
                    all_preds.append(pred)
                
                pred = torch.cat(all_preds, dim=0)
            else:
                # Standard forward pass (no augmentation during eval)
                if self.use_amp:
                    with autocast():
                        model_output = self.model.forward(X, dayIdx)
                else:
                    model_output = self.model.forward(X, dayIdx)
                
                # Get phone logits (handle both output formats)
                if isinstance(model_output, dict):
                    pred = model_output['phone_logits']
                else:
                    pred = model_output
            
            # CTC loss requires float32 (doesn't support FP16)
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
                
                matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
                total_edit_distance += matcher.distance()
                total_seq_length += len(trueSeq)
        
        n_batches = min(len(loader), max_batches) if max_batches else len(loader)
        avg_loss = np.sum(all_loss) / max(n_batches, 1)
        per = total_edit_distance / total_seq_length if total_seq_length > 0 else 1.0
        
        return avg_loss, per
