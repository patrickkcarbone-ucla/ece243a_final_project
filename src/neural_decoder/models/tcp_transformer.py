"""
TCP (Temporal Coarticulation Pyramid) Transformer for neural speech decoding.

Extends the Time-Masked Transformer with multi-level prediction heads:
- MonophoneHead (Level 1): Early layer intermediate CTC supervision
- DiphoneHead (Level 2): Middle layer diphone prediction with marginalization
- ContextHead (Level 3): Final layer context-aware prediction

Key features:
- Configurable tap layers for multi-level outputs
- Optional learned gating network for adaptive level weighting
- Gradient highway connection for stable training
- Compatible with TCP curriculum and Feghhi schedules
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class T5RelativePositionBias(nn.Module):
    """
    T5-style relative positional embeddings.

    Learns a bias table indexed by relative distance between query and key positions.
    These biases are added to attention logits before softmax.
    """

    def __init__(self, num_heads, max_rel_pos=64):
        super().__init__()
        self.num_heads = num_heads
        self.max_rel_pos = max_rel_pos

        # Bias table: (num_heads, 2 * max_rel_pos + 1)
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_rel_pos + 1)
        )
        nn.init.normal_(self.relative_bias, std=0.02)

    def forward(self, seq_len):
        """Compute relative position bias matrix."""
        positions = torch.arange(seq_len, device=self.relative_bias.device)
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_pos = relative_pos.clamp(-self.max_rel_pos, self.max_rel_pos)
        relative_pos = relative_pos + self.max_rel_pos
        bias = self.relative_bias[:, relative_pos]
        return bias.unsqueeze(0)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with T5 relative position bias."""

    def __init__(
        self, hidden_dim, num_heads, attn_dropout=0.1, max_rel_pos=64
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.rel_pos_bias = T5RelativePositionBias(num_heads, max_rel_pos)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_bias = self.rel_pos_bias(T)
        attn = attn + rel_bias

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)

        return out


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with causal self-attention and FFN."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        ffn_dim,
        dropout=0.35,
        attn_dropout=0.1,
        max_rel_pos=64,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(
            hidden_dim, num_heads, attn_dropout, max_rel_pos
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# TCP-Specific Components
# =============================================================================

class MonophoneHead(nn.Module):
    """
    Simple LN -> Linear head for intermediate CTC supervision (Level 1).

    Provides early-layer regularization by requiring lower Transformer layers
    to produce valid phoneme predictions.

    Args:
        hidden_dim: Transformer hidden dimension
        n_classes: Number of phoneme classes (40)
    """

    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        """
        Args:
            x: [B, T, hidden_dim] intermediate representation
        Returns:
            logits: [B, T, n_classes + 1]
        """
        return self.proj(self.ln(x))


class DiphoneHead(nn.Module):
    """
    LN -> Linear head for diphone prediction with marginalization (Level 2).

    Captures coarticulation dynamics by predicting (prev_phoneme, curr_phoneme) pairs.
    Includes marginalization to convert diphone logits to monophone space for fusion.

    Args:
        hidden_dim: Transformer hidden dimension
        n_classes: Number of phoneme classes (40)
        n_diphones: Number of diphone classes (40 * 40 = 1600)
    """

    def __init__(self, hidden_dim, n_classes=40, n_diphones=1600):
        super().__init__()
        self.n_classes = n_classes
        self.n_diphones = n_diphones

        self.ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, n_diphones + 1)  # +1 for CTC blank

        # Store raw logits for diphone CTC loss
        self.raw_logits = None

    def forward(self, x):
        """
        Args:
            x: [B, T, hidden_dim] intermediate representation
        Returns:
            marginalized: [B, T, n_classes + 1] marginalized monophone logits
        """
        logits = self.proj(self.ln(x))  # [B, T, n_diphones + 1]
        self.raw_logits = logits  # Store for diphone CTC loss

        # Marginalize to monophone space for gated fusion
        return self.marginalize(logits)

    def marginalize(self, logits):
        """
        Convert diphone logits to monophone probabilities.

        For diphone (p1, p2), marginalize over p1 to get P(p2).

        Args:
            logits: [B, T, n_diphones + 1] raw diphone logits
        Returns:
            mono_logits: [B, T, n_classes + 1] marginalized monophone logits
        """
        B, T, D = logits.shape

        # Separate blank and diphone logits
        blank_logit = logits[:, :, 0:1]  # [B, T, 1]
        diphone_logits = logits[:, :, 1:]  # [B, T, n_diphones]

        # Apply softmax over full diphone space
        probs = F.softmax(logits, dim=-1)
        diphone_probs = probs[:, :, 1:]  # [B, T, n_diphones]

        # Reshape to [B, T, n_classes, n_classes] where [i, j] = P(prev=i, curr=j)
        diphone_probs = diphone_probs.view(
            B, T, self.n_classes, self.n_classes
        )

        # Sum over first phoneme (prev) to get marginal P(curr)
        mono_probs = diphone_probs.sum(dim=2)  # [B, T, n_classes]

        # Convert back to logits (add small epsilon for numerical stability)
        mono_logits = torch.log(mono_probs + 1e-8)

        # Append blank (use original blank probability)
        blank_prob = probs[:, :, 0:1]
        blank_logit_out = torch.log(blank_prob + 1e-8)

        # Combine: [blank, mono_1, ..., mono_40]
        return torch.cat([blank_logit_out, mono_logits], dim=-1)


class ContextHead(nn.Module):
    """
    Two-layer MLP for context-aware final prediction (Level 3).

    Leverages full temporal context from the final layer for phoneme prediction.
    Uses deeper MLP than Level 1 since final representation is richest.

    Args:
        hidden_dim: Transformer hidden dimension
        n_classes: Number of phoneme classes (40)
    """

    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes + 1),  # +1 for CTC blank
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, hidden_dim] final layer representation
        Returns:
            logits: [B, T, n_classes + 1]
        """
        return self.mlp(self.ln(x))


class GatingNetwork(nn.Module):
    """
    Adaptive weighting of three prediction levels.

    Learns to allocate trust across abstraction levels based on:
    - Context summary from final layer
    - Current mask ratio (optional, for masking-adaptive behavior)

    Args:
        hidden_dim: Transformer hidden dimension
        use_mask_ratio: Whether to include mask ratio as input
    """

    def __init__(self, hidden_dim, use_mask_ratio=True):
        super().__init__()
        self.use_mask_ratio = use_mask_ratio

        input_dim = hidden_dim + 1 if use_mask_ratio else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # 3 levels: mono, diphone, context
        )

    def forward(self, context_summary, mask_ratio=0.0):
        """
        Args:
            context_summary: [B, hidden_dim] temporal mean of final layer
            mask_ratio: float or [B, 1] fraction of masked patches
        Returns:
            gates: [B, 3] softmax weights for (mono, diphone, context)
        """
        B = context_summary.shape[0]
        device = context_summary.device

        if self.use_mask_ratio:
            # Broadcast mask_ratio to [B, 1]
            if isinstance(mask_ratio, float):
                mask_ratio_tensor = torch.full(
                    (B, 1), mask_ratio, device=device
                )
            else:
                mask_ratio_tensor = mask_ratio.view(B, 1)

            x = torch.cat([context_summary, mask_ratio_tensor], dim=-1)
        else:
            x = context_summary

        logits = self.mlp(x)
        return F.softmax(logits, dim=-1)


class GradientHighway(nn.Module):
    """
    Residual pathway from early layer to upper layers.

    Provides stable gradient flow to lower layers by adding a scaled
    projection of early representations to upper layers.

    The input is detached during forward to prevent conflicting gradients,
    but the highway output still provides a "memory" of early representations.

    Args:
        hidden_dim: Transformer hidden dimension
        scale: Scaling factor for highway output (default: 0.1)
    """

    def __init__(self, hidden_dim, scale=0.1):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: [B, T, hidden_dim] early layer representation (should be detached)
        Returns:
            highway_out: [B, T, hidden_dim] scaled projection
        """
        return self.scale * self.proj(x)


# =============================================================================
# Main TCP Transformer
# =============================================================================


class TCPTransformer(nn.Module):
    """
    TCP (Temporal Coarticulation Pyramid) Transformer for neural speech decoding.

    Extends the Time-Masked Transformer with multi-level prediction heads:
    - Level 1 (mono_layer): Monophone intermediate CTC
    - Level 2 (diphone_layer): Diphone prediction with marginalization
    - Level 3 (final layer): Context-aware monophone prediction

    Features:
    - Configurable tap layers for multi-level outputs
    - Optional learned gating for adaptive level weighting
    - Gradient highway for stable training
    - Backward compatible: set use_tcp=False for standard Transformer behavior

    Args:
        neural_dim: Input feature dimension (256 for 6v channels)
        n_classes: Number of phoneme classes (40)
        hidden_dim: Transformer hidden dimension (384)
        num_layers: Number of Transformer blocks (5)
        num_heads: Number of attention heads (6)
        ffn_dim: Feed-forward network dimension (1536)
        patch_len: Number of time bins per patch (5 = 100ms)
        patch_stride: Stride between patches (5 = non-overlapping)
        dropout: FFN dropout rate (0.35)
        attn_dropout: Attention dropout rate (0.1)
        input_dropout: Input dropout rate (0.2)
        max_rel_pos: Maximum relative position for T5 bias (64)
        num_masks: Number of time masks to apply (20)
        max_mask_fraction: Maximum fraction of patches per mask (0.075)
        use_tcp: Enable TCP multi-level prediction (default: False)
        mono_layer: Layer index for monophone head (0-indexed, default: 1)
        diphone_layer: Layer index for diphone head (0-indexed, default: 2)
        use_gating: Use learned gating vs fixed weights (default: True)
        use_highway: Enable gradient highway (default: True)
        fixed_gate_weights: Weights when gating disabled (default: (0.33, 0.33, 0.34))
        n_diphones: Number of diphone classes (default: 1600)
        nDays: Unused, kept for config compatibility
        device: Device to run on
    """

    def __init__(
        self,
        neural_dim=256,
        n_classes=40,
        hidden_dim=384,
        num_layers=5,
        num_heads=6,
        ffn_dim=1536,
        patch_len=5,
        patch_stride=5,
        dropout=0.35,
        attn_dropout=0.1, # what is this parameter?
        input_dropout=0.2,
        max_rel_pos=64,
        num_masks=20,
        max_mask_fraction=0.075,
        # TCP-specific parameters
        use_tcp=False,
        mono_layer=1,
        diphone_layer=2,
        use_gating=True,
        use_highway=True,
        fixed_gate_weights=(0.33, 0.33, 0.34),
        n_diphones=1600,
        # Compatibility
        nDays=24,
        device="cuda",
    ):
        super().__init__()

        # Base parameters
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

        # For compatibility with CTCTrainer's output length calculation
        self.kernelLen = patch_len
        self.strideLen = patch_stride

        # Patch embedding
        patch_input_dim = neural_dim * patch_len
        self.patch_embed = nn.Linear(patch_input_dim, hidden_dim)

        # Input dropout
        self.input_dropout = nn.Dropout(input_dropout)

        # Learnable mask token for time-masking
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    max_rel_pos=max_rel_pos,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.ln_final = nn.LayerNorm(hidden_dim)

        # Standard output head (used when use_tcp=False)
        self.fc_out = nn.Linear(hidden_dim, n_classes + 1)

        # TCP components (only initialized if use_tcp=True)
        if self.use_tcp:
            self.mono_head = MonophoneHead(hidden_dim, n_classes)
            self.diphone_head = DiphoneHead(hidden_dim, n_classes, n_diphones)
            self.context_head = ContextHead(hidden_dim, n_classes)

            if self.use_gating:
                self.gating = GatingNetwork(hidden_dim, use_mask_ratio=True)

            if self.use_highway:
                self.highway = GradientHighway(hidden_dim, scale=0.1)

        # Track mask ratio during training (for gating network)
        self._current_mask_ratio = 0.0

    def _form_patches(self, x):
        """Form non-overlapping patches from input sequence."""
        B, T, D = x.shape

        num_patches = (T - self.patch_len) // self.patch_stride + 1
        valid_len = (num_patches - 1) * self.patch_stride + self.patch_len
        x = x[:, :valid_len, :]

        x = x.transpose(1, 2)
        x = x.unfold(2, self.patch_len, self.patch_stride)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, num_patches, -1)

        return x

    def _apply_time_masking(self, x):
        """Apply time-masking augmentation during training."""
        if not self.training or self.num_masks == 0:
            self._current_mask_ratio = 0.0
            return x

        B, T, D = x.shape
        mask = torch.ones(B, T, device=x.device, dtype=torch.bool)

        for _ in range(self.num_masks):
            max_mask_len = max(1, int(T * self.max_mask_fraction))
            mask_lens = torch.randint(
                1, max_mask_len + 1, (B,), device=x.device
            )

            max_starts = T - mask_lens
            max_starts = max_starts.clamp(min=0)
            starts = (torch.rand(B, device=x.device) * (max_starts + 1)).long()

            for b in range(B):
                start = starts[b].item()
                end = min(start + mask_lens[b].item(), T)
                mask[b, start:end] = False

        # Compute mask ratio for gating network
        self._current_mask_ratio = 1.0 - mask.float().mean().item()

        mask_expanded = mask.unsqueeze(-1)
        x = torch.where(mask_expanded, x, self.mask_token.expand_as(x))

        return x

    def _apply_gates(self, z_mono, z_diphone, z_context, gates):
        """
        Apply learned gates to combine three prediction levels.

        Args:
            z_mono: [B, T, n_classes + 1] monophone logits
            z_diphone: [B, T, n_classes + 1] marginalized diphone logits
            z_context: [B, T, n_classes + 1] context head logits
            gates: [B, 3] gate weights
        Returns:
            z_fused: [B, T, n_classes + 1] gated fusion
        """
        # Expand gates for broadcasting: [B, 1, 1]
        g_mono = gates[:, 0:1].unsqueeze(-1)
        g_diphone = gates[:, 1:2].unsqueeze(-1)
        g_context = gates[:, 2:3].unsqueeze(-1)

        z_fused = (
            g_mono * z_mono + g_diphone * z_diphone + g_context * z_context
        )
        return z_fused

    def _apply_fixed_gates(self, z_mono, z_diphone, z_context):
        """Apply fixed gate weights to combine three prediction levels."""
        w_mono, w_diphone, w_context = self.fixed_gate_weights
        return w_mono * z_mono + w_diphone * z_diphone + w_context * z_context

    def forward(self, neuralInput, dayIdx=None):
        """
        Forward pass.

        Args:
            neuralInput: [B, T, neural_dim] raw neural features
            dayIdx: Unused, kept for API compatibility

        Returns:
            If use_tcp=False:
                logits: [B, num_patches, n_classes + 1]
            If use_tcp=True:
                dict with keys:
                    - 'phone_logits': [B, T', n_classes + 1] (gated fusion)
                    - 'mono_logits': [B, T', n_classes + 1]
                    - 'diphone_logits': [B, T', n_diphones + 1] (raw, for CTC)
                    - 'context_logits': [B, T', n_classes + 1]
                    - 'gates': [B, 3] or None
                    - 'mask_ratio': float
        """
        x = neuralInput

        # Form patches
        x = self._form_patches(x)

        # Project to hidden dimension
        x = self.patch_embed(x)

        # Input dropout
        x = self.input_dropout(x)

        # Apply time-masking (training only)
        x = self._apply_time_masking(x)

        # Process through Transformer blocks with TCP tap points
        intermediate_outputs = {}
        highway_input = None

        for i, block in enumerate(self.blocks):
            # TCP: Apply highway BEFORE block for upper layers
            # This injects early-layer information into upper blocks' inputs
            if self.use_tcp and self.use_highway and highway_input is not None:
                if i >= self.mono_layer + 2:
                    x = x + self.highway(highway_input)

            # Run the Transformer block
            x = block(x)

            # TCP: Capture tap points AFTER block
            if self.use_tcp:
                if i == self.mono_layer:
                    intermediate_outputs["mono_hidden"] = x
                    if self.use_highway:
                        highway_input = x.detach()  # Detach for highway

                if i == self.diphone_layer:
                    intermediate_outputs["diphone_hidden"] = x

        # Final layer norm
        x = self.ln_final(x)

        # Non-TCP mode: standard output
        if not self.use_tcp:
            return self.fc_out(x)

        # TCP mode: compute all head outputs
        z_mono = self.mono_head(intermediate_outputs["mono_hidden"])
        z_diphone_marginalized = self.diphone_head(
            intermediate_outputs["diphone_hidden"]
        )
        z_context = self.context_head(x)

        # Gated fusion
        if self.use_gating:
            context_summary = x.mean(dim=1)  # [B, hidden_dim]
            gates = self.gating(context_summary, self._current_mask_ratio)
            z_fused = self._apply_gates(
                z_mono, z_diphone_marginalized, z_context, gates
            )
        else:
            gates = None
            z_fused = self._apply_fixed_gates(
                z_mono, z_diphone_marginalized, z_context
            )

        return {
            "phone_logits": z_fused,
            "mono_logits": z_mono,
            "diphone_logits": self.diphone_head.raw_logits,  # [B, T, n_diphones + 1]
            "context_logits": z_context,
            "gates": gates,
            "mask_ratio": self._current_mask_ratio,
        }
