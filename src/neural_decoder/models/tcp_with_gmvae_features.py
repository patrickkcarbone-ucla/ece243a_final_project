"""
TCP Transformer with Frozen GMVAE Features.

This model uses a pre-trained GMVAE as a frozen feature extractor.
The GMVAE cluster probabilities are concatenated to the patched
neural data before feeding to the TCP Transformer.

Architecture:
    Neural Input [B, T, 256] (20ms resolution)
           |
       Patching (5 bins = 100ms)
           |
    [B, T', 1280] (256 * 5 flattened)
           |
       Frozen GMVAE Encoder
           |
       cluster_probs [B, T', K]
           |
       concat [patches, cluster_probs]
           |
    [B, T', 1280 + K]
           |
       TCP Transformer
           |
    Output: phoneme logits

Key features:
- GMVAE is frozen (no gradient updates)
- Cluster probs are extra "virtual channels"
- Full TCP multi-level prediction (mono, diphone, context)
- Configurable GMVAE weights path for selecting different trained models
"""

import os
import math
import torch
from torch import nn
import torch.nn.functional as F

from neural_decoder.models.tcp_transformer import (
    TransformerBlock,
    MonophoneHead,
    DiphoneHead,
    ContextHead,
    GatingNetwork,
    GradientHighway,
)
from neural_decoder.models.gmvae import StandaloneGMVAE


class TCPWithGMVAEFeatures(nn.Module):
    """
    TCP Transformer with frozen GMVAE features.

    Uses a pre-trained GMVAE to extract cluster probabilities from
    neural data, then concatenates these to the patched input before
    feeding to a TCP Transformer.

    The GMVAE is frozen during training - only TCP parameters are updated.

    Args:
        neural_dim: Raw neural channel dimension (256 for 6v)
        n_classes: Number of phoneme classes (40)
        hidden_dim: TCP Transformer hidden dimension
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension
        patch_len: Number of time bins per patch (5 = 100ms)
        patch_stride: Stride between patches (5 = non-overlapping)
        dropout: FFN dropout rate
        attn_dropout: Attention dropout rate
        input_dropout: Input dropout rate
        max_rel_pos: Maximum relative position for T5 bias
        num_masks: Number of time masks to apply
        max_mask_fraction: Maximum fraction of patches per mask
        use_tcp: Enable TCP multi-level prediction
        mono_layer: Layer index for monophone head
        diphone_layer: Layer index for diphone head
        use_gating: Use learned gating vs fixed weights
        use_highway: Enable gradient highway
        fixed_gate_weights: Weights when gating disabled
        n_diphones: Number of diphone classes
        gmvae_weights_path: Path to pre-trained GMVAE weights (required)
        gmvae_n_clusters: Number of GMVAE clusters (must match saved model)
        gmvae_hidden_dim: GMVAE hidden dimension (must match saved model)
        gmvae_z_dim: GMVAE latent dimension (must match saved model)
        nDays: Unused, for config compatibility
        device: Device to run on
    """

    def __init__(
        self,
        neural_dim=256,
        n_classes=40,
        hidden_dim=896,
        num_layers=10,
        num_heads=8,
        ffn_dim=3584,
        patch_len=5,
        patch_stride=5,
        dropout=0.4,
        attn_dropout=0.1,
        input_dropout=0.2,
        max_rel_pos=64,
        num_masks=25,
        max_mask_fraction=0.08,
        # TCP parameters
        use_tcp=True,
        mono_layer=4,
        diphone_layer=7,
        use_gating=True,
        use_highway=True,
        fixed_gate_weights=(0.33, 0.33, 0.34),
        n_diphones=1600,
        # GMVAE parameters
        gmvae_weights_path=None,
        gmvae_n_clusters=40,
        gmvae_hidden_dim=512,
        gmvae_z_dim=64,
        # Compatibility
        nDays=24,
        device="cuda",
    ):
        super().__init__()

        # Validate GMVAE weights path
        if gmvae_weights_path is None:
            raise ValueError(
                "gmvae_weights_path is required. Please provide path to pre-trained GMVAE weights."
            )

        # Store parameters
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.device = device
        self.num_masks = num_masks
        self.max_mask_fraction = max_mask_fraction

        # TCP parameters
        self.use_tcp = use_tcp
        self.mono_layer = mono_layer
        self.diphone_layer = diphone_layer
        self.use_gating = use_gating
        self.use_highway = use_highway
        self.fixed_gate_weights = fixed_gate_weights
        self.n_diphones = n_diphones

        # GMVAE parameters
        self.gmvae_n_clusters = gmvae_n_clusters

        # For CTC length calculation (compatible with trainer)
        self.kernelLen = patch_len
        self.strideLen = patch_stride

        # =====================================================================
        # Frozen GMVAE Feature Extractor
        # =====================================================================

        self.gmvae = StandaloneGMVAE(
            neural_dim=neural_dim,
            patch_len=patch_len,
            patch_stride=patch_stride,
            hidden_dim=gmvae_hidden_dim,
            z_dim=gmvae_z_dim,
            n_clusters=gmvae_n_clusters,
            n_phonemes=n_classes,
            device=device,
        )

        # Load pre-trained weights
        self._load_gmvae_weights(gmvae_weights_path)

        # Freeze GMVAE parameters
        self._freeze_gmvae()

        # =====================================================================
        # TCP Transformer (modified input dimension)
        # =====================================================================

        # Input dimension: patched neural + cluster probs
        # patches are [B, T', patch_dim] where patch_dim = neural_dim * patch_len
        # cluster_probs are [B, T', n_clusters]
        # concatenated: [B, T', patch_dim + n_clusters]
        patch_dim = neural_dim * patch_len
        enriched_dim = patch_dim + gmvae_n_clusters

        # Patch embedding (from enriched input)
        self.patch_embed = nn.Linear(enriched_dim, hidden_dim)

        # Input dropout
        self.input_dropout = nn.Dropout(input_dropout)

        # Learnable mask token for time-masking
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                max_rel_pos=max_rel_pos,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(hidden_dim)

        # Standard output head (used when use_tcp=False)
        self.fc_out = nn.Linear(hidden_dim, n_classes + 1)

        # TCP components
        if self.use_tcp:
            self.mono_head = MonophoneHead(hidden_dim, n_classes)
            self.diphone_head = DiphoneHead(hidden_dim, n_classes, n_diphones)
            self.context_head = ContextHead(hidden_dim, n_classes)

            if self.use_gating:
                self.gating = GatingNetwork(hidden_dim, use_mask_ratio=True)

            if self.use_highway:
                self.highway = GradientHighway(hidden_dim, scale=0.1)

        # Track mask ratio during training
        self._current_mask_ratio = 0.0

    def _load_gmvae_weights(self, weights_path):
        """Load pre-trained GMVAE weights."""
        # Handle relative paths with Hydra
        if not os.path.isabs(weights_path):
            try:
                import hydra
                orig_cwd = hydra.utils.get_original_cwd()
                weights_path = os.path.join(orig_cwd, weights_path)
            except Exception:
                pass

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"GMVAE weights not found at: {weights_path}\n"
                f"Please train a GMVAE first using the gmvae/standalone experiment."
            )

        print(f"Loading GMVAE weights from: {weights_path}")

        # Load state dict
        state_dict = torch.load(weights_path, map_location='cpu')

        # Handle potential key mismatches (e.g., if saved with DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load only GMVAE-related keys (exclude eval_head which we don't need)
        gmvae_keys = [k for k in state_dict.keys() if not k.startswith('eval_head')]
        gmvae_state = {k: v for k, v in state_dict.items() if k in gmvae_keys}

        # Load with strict=False to handle eval_head mismatch
        missing, unexpected = self.gmvae.load_state_dict(gmvae_state, strict=False)

        if missing:
            # Filter out expected missing keys (eval_head)
            missing = [k for k in missing if not k.startswith('eval_head')]
            if missing:
                print(f"Warning: Missing GMVAE keys: {missing}")

        print(f"Successfully loaded GMVAE weights ({len(gmvae_state)} parameters)")

    def _freeze_gmvae(self):
        """Freeze all GMVAE parameters."""
        for param in self.gmvae.parameters():
            param.requires_grad = False

        # Set to eval mode permanently
        self.gmvae.eval()

        print("GMVAE parameters frozen (no gradient updates)")

    def _apply_time_masking(self, x):
        """Apply time-masking augmentation during training."""
        if not self.training or self.num_masks == 0:
            self._current_mask_ratio = 0.0
            return x

        B, T, D = x.shape
        mask = torch.ones(B, T, device=x.device, dtype=torch.bool)

        for _ in range(self.num_masks):
            max_mask_len = max(1, int(T * self.max_mask_fraction))
            mask_lens = torch.randint(1, max_mask_len + 1, (B,), device=x.device)

            max_starts = T - mask_lens
            max_starts = max_starts.clamp(min=0)
            starts = (torch.rand(B, device=x.device) * (max_starts + 1)).long()

            for b in range(B):
                start = starts[b].item()
                end = min(start + mask_lens[b].item(), T)
                mask[b, start:end] = False

        self._current_mask_ratio = 1.0 - mask.float().mean().item()

        mask_expanded = mask.unsqueeze(-1)
        x = torch.where(mask_expanded, x, self.mask_token.expand_as(x))

        return x

    def _apply_gates(self, z_mono, z_diphone, z_context, gates):
        """Apply learned gates to combine three prediction levels."""
        g_mono = gates[:, 0:1].unsqueeze(-1)
        g_diphone = gates[:, 1:2].unsqueeze(-1)
        g_context = gates[:, 2:3].unsqueeze(-1)

        return g_mono * z_mono + g_diphone * z_diphone + g_context * z_context

    def _apply_fixed_gates(self, z_mono, z_diphone, z_context):
        """Apply fixed gate weights to combine three prediction levels."""
        w_mono, w_diphone, w_context = self.fixed_gate_weights
        return w_mono * z_mono + w_diphone * z_diphone + w_context * z_context

    def forward(self, neuralInput, dayIdx=None):
        """
        Forward pass.

        Args:
            neuralInput: [B, T, neural_dim] raw neural features (20ms resolution)
            dayIdx: Unused, kept for API compatibility

        Returns:
            dict with:
                - phone_logits: [B, T', n_classes + 1] final output (gated fusion)
                - mono_logits: [B, T', n_classes + 1] (if use_tcp)
                - diphone_logits: [B, T', n_diphones + 1] raw (if use_tcp)
                - context_logits: [B, T', n_classes + 1] (if use_tcp)
                - gates: [B, 3] TCP gate values (if use_tcp and use_gating)
                - cluster_probs: [B, T', n_clusters] GMVAE cluster probs
                - mask_ratio: float
        """
        B, T, D = neuralInput.shape

        # =====================================================================
        # 1. Get cluster probabilities from frozen GMVAE
        # =====================================================================

        with torch.no_grad():
            # Ensure GMVAE is in eval mode
            self.gmvae.eval()

            # Forward through GMVAE (only need cluster probs)
            gmvae_out = self.gmvae(neuralInput)
            cluster_probs = gmvae_out['prob_cat']  # [B, T', n_clusters]
            patches = gmvae_out['patches']  # [B, T', patch_dim]

        # =====================================================================
        # 2. Concatenate patches with cluster probabilities
        # =====================================================================

        # patches: [B, T', 1280]
        # cluster_probs: [B, T', 40]
        # enriched: [B, T', 1320]
        enriched = torch.cat([patches, cluster_probs], dim=-1)

        # =====================================================================
        # 3. TCP Transformer
        # =====================================================================

        # Project to hidden dimension
        x = self.patch_embed(enriched)

        # Input dropout
        x = self.input_dropout(x)

        # Apply time-masking (training only)
        x = self._apply_time_masking(x)

        # Process through Transformer blocks with TCP tap points
        intermediate_outputs = {}
        highway_input = None

        for i, block in enumerate(self.blocks):
            # Apply highway BEFORE block for upper layers
            if self.use_tcp and self.use_highway and highway_input is not None:
                if i >= self.mono_layer + 2:
                    x = x + self.highway(highway_input)

            # Run the Transformer block
            x = block(x)

            # Capture tap points AFTER block
            if self.use_tcp:
                if i == self.mono_layer:
                    intermediate_outputs["mono_hidden"] = x
                    if self.use_highway:
                        highway_input = x.detach()

                if i == self.diphone_layer:
                    intermediate_outputs["diphone_hidden"] = x

        # Final layer norm
        x = self.ln_final(x)

        # Non-TCP mode: standard output
        if not self.use_tcp:
            return {
                'phone_logits': self.fc_out(x),
                'cluster_probs': cluster_probs,
                'mask_ratio': self._current_mask_ratio,
            }

        # =====================================================================
        # 4. TCP heads and gated fusion
        # =====================================================================

        z_mono = self.mono_head(intermediate_outputs["mono_hidden"])
        z_diphone_marginalized = self.diphone_head(intermediate_outputs["diphone_hidden"])
        z_context = self.context_head(x)

        if self.use_gating:
            context_summary = x.mean(dim=1)
            gates = self.gating(context_summary, self._current_mask_ratio)
            z_fused = self._apply_gates(z_mono, z_diphone_marginalized, z_context, gates)
        else:
            gates = None
            z_fused = self._apply_fixed_gates(z_mono, z_diphone_marginalized, z_context)

        return {
            'phone_logits': z_fused,
            'mono_logits': z_mono,
            'diphone_logits': self.diphone_head.raw_logits,
            'context_logits': z_context,
            'gates': gates,
            'cluster_probs': cluster_probs,
            'mask_ratio': self._current_mask_ratio,
        }

    def get_trainable_parameters(self):
        """Get only trainable (non-frozen) parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable_params(self):
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_frozen_params(self):
        """Get count of frozen parameters (GMVAE)."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)