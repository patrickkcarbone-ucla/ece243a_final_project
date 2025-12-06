"""
Time-Masked Transformer for neural speech decoding.

Based on Feghhi et al. (2025) with adaptations for the Brain-to-Text benchmark.
Key features:
- Non-overlapping patch embedding (5 time bins = 100ms)
- T5-style relative positional embeddings
- Causal (unidirectional) attention for streaming compatibility
- Learnable mask token for time-masking augmentation
- NO day-specific layers (unlike baseline GRU)
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
        # Indices: -max_rel_pos to +max_rel_pos
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_rel_pos + 1)
        )
        nn.init.normal_(self.relative_bias, std=0.02)
    
    def forward(self, seq_len):
        """
        Compute relative position bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            bias: [1, num_heads, seq_len, seq_len] bias to add to attention logits
        """
        # Create position indices
        positions = torch.arange(seq_len, device=self.relative_bias.device)
        # Compute relative distances: query_pos - key_pos
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
        
        # Clamp to valid range and shift to positive indices
        relative_pos = relative_pos.clamp(-self.max_rel_pos, self.max_rel_pos)
        relative_pos = relative_pos + self.max_rel_pos  # Now in [0, 2*max_rel_pos]
        
        # Gather biases for each head
        # relative_bias: [num_heads, 2*max_rel_pos+1]
        # relative_pos: [seq_len, seq_len]
        bias = self.relative_bias[:, relative_pos]  # [num_heads, seq_len, seq_len]
        
        return bias.unsqueeze(0)  # [1, num_heads, seq_len, seq_len]


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with T5 relative position bias.
    """
    
    def __init__(self, hidden_dim, num_heads, attn_dropout=0.1, max_rel_pos=64):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.rel_pos_bias = T5RelativePositionBias(num_heads, max_rel_pos)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, hidden_dim]
        Returns:
            out: [B, T, hidden_dim]
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        
        # Add relative position bias
        rel_bias = self.rel_pos_bias(T)  # [1, num_heads, T, T]
        attn = attn + rel_bias
        
        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), 
            diagonal=1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with causal self-attention and FFN.
    """
    
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
        self.attn = CausalSelfAttention(hidden_dim, num_heads, attn_dropout, max_rel_pos)
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
        # Pre-norm attention
        x = x + self.dropout1(self.attn(self.ln1(x)))
        # Pre-norm FFN
        x = x + self.ffn(self.ln2(x))
        return x


class TimeMaskedTransformer(nn.Module):
    """
    Time-Masked Transformer decoder for neural speech decoding.
    
    Processes neural input by:
    1. Forming non-overlapping patches (5 time bins = 100ms)
    2. Projecting to hidden dimension
    3. Applying time-masking (training only)
    4. Processing through causal Transformer layers
    5. Projecting to phoneme classes for CTC loss
    
    NOTE: Unlike the GRU baseline, this model does NOT use day-specific
    transformation layers, following Feghhi et al. (2025).
    
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
        use_diphone: Enable diphone auxiliary output head
        n_diphones: Number of diphone classes (set from vocab)
        use_intermediate_ctc: Enable intermediate CTC loss from middle layer
        intermediate_layer: Layer index for intermediate output (0-indexed)
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
        attn_dropout=0.1,
        input_dropout=0.2,
        max_rel_pos=64,
        num_masks=20,
        max_mask_fraction=0.075,
        use_diphone=False,
        n_diphones=1600,
        use_intermediate_ctc=False,
        intermediate_layer=2,
        nDays=24,  # Unused, kept for config compatibility
        device="cuda",
    ):
        super().__init__()
        
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.device = device
        self.num_masks = num_masks
        self.max_mask_fraction = max_mask_fraction
        
        # Part 2 features
        self.use_diphone = use_diphone
        self.n_diphones = n_diphones
        self.use_intermediate_ctc = use_intermediate_ctc
        self.intermediate_layer = intermediate_layer
        
        # For compatibility with CTCTrainer's output length calculation
        # The trainer uses (X_len - kernelLen) / strideLen
        # For transformer: output_len = (input_len - patch_len) / patch_stride + 1
        # To match: kernelLen = patch_len, strideLen = patch_stride
        self.kernelLen = patch_len
        self.strideLen = patch_stride
        
        # NOTE: Unlike the GRU baseline, the Transformer does NOT use day-specific
        # transformation layers. This follows Feghhi et al. (2025).
        
        # Patch embedding: flatten patch_len time steps and project
        patch_input_dim = neural_dim * patch_len
        self.patch_embed = nn.Linear(patch_input_dim, hidden_dim)
        
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
        
        # Final layer norm (pre-norm architecture)
        self.ln_final = nn.LayerNorm(hidden_dim)
        
        # Output projection to phoneme classes + blank for CTC
        self.fc_out = nn.Linear(hidden_dim, n_classes + 1)
        
        # Diphone output head (optional)
        if self.use_diphone:
            self.diphone_head = nn.Linear(hidden_dim, n_diphones + 1)  # +1 for blank
        
        # Intermediate CTC projection (optional)
        if self.use_intermediate_ctc:
            self.intermediate_ln = nn.LayerNorm(hidden_dim)
            self.intermediate_proj = nn.Linear(hidden_dim, n_classes + 1)
    
    def _form_patches(self, x):
        """
        Form non-overlapping patches from input sequence.
        
        Args:
            x: [B, T, neural_dim]
        Returns:
            patches: [B, num_patches, patch_len * neural_dim]
        """
        B, T, D = x.shape
        
        # Truncate to multiple of patch_stride
        num_patches = (T - self.patch_len) // self.patch_stride + 1
        valid_len = (num_patches - 1) * self.patch_stride + self.patch_len
        x = x[:, :valid_len, :]
        
        # Use unfold to create patches
        # Reshape to [B, D, T] for unfold, then back
        x = x.transpose(1, 2)  # [B, D, T]
        x = x.unfold(2, self.patch_len, self.patch_stride)  # [B, D, num_patches, patch_len]
        x = x.permute(0, 2, 3, 1)  # [B, num_patches, patch_len, D]
        x = x.reshape(B, num_patches, -1)  # [B, num_patches, patch_len * D]
        
        return x
    
    def _apply_time_masking(self, x):
        """
        Apply time-masking augmentation during training.
        
        Replaces random contiguous spans with learnable mask token.
        Expected to mask ~53% of patches on average.
        
        Args:
            x: [B, num_patches, hidden_dim] (after patch embedding)
        Returns:
            x: [B, num_patches, hidden_dim] with masked regions
        """
        if not self.training or self.num_masks == 0:
            return x
        
        B, T, D = x.shape
        mask = torch.ones(B, T, device=x.device, dtype=torch.bool)
        
        for _ in range(self.num_masks):
            # Random mask length for each sample
            max_mask_len = max(1, int(T * self.max_mask_fraction))
            mask_lens = torch.randint(1, max_mask_len + 1, (B,), device=x.device)
            
            # Random start positions
            max_starts = T - mask_lens
            max_starts = max_starts.clamp(min=0)
            starts = (torch.rand(B, device=x.device) * (max_starts + 1)).long()
            
            # Apply masks
            for b in range(B):
                start = starts[b].item()
                end = min(start + mask_lens[b].item(), T)
                mask[b, start:end] = False
        
        # Replace masked positions with mask token
        mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
        x = torch.where(mask_expanded, x, self.mask_token.expand_as(x))
        
        return x
    
    def forward(self, neuralInput, dayIdx=None):
        """
        Forward pass.
        
        Args:
            neuralInput: [B, T, neural_dim] raw neural features
            dayIdx: Unused, kept for API compatibility with GRU trainer
            
        Returns:
            If use_diphone=False and use_intermediate_ctc=False:
                logits: [B, num_patches, n_classes + 1]
            Otherwise:
                dict with keys:
                    - 'phone_logits': [B, T', n_classes + 1]
                    - 'diphone_logits': [B, T', n_diphones + 1] (if use_diphone)
                    - 'intermediate_logits': [B, T', n_classes + 1] (if use_intermediate_ctc)
        """
        # NOTE: Unlike GRU, Transformer does NOT apply day-specific transformation
        x = neuralInput
        
        # Form patches
        x = self._form_patches(x)  # [B, num_patches, patch_len * neural_dim]
        
        # Project to hidden dimension
        x = self.patch_embed(x)  # [B, num_patches, hidden_dim]
        
        # Input dropout
        x = self.input_dropout(x)
        
        # Apply time-masking (training only)
        x = self._apply_time_masking(x)
        
        # Transformer blocks with optional intermediate capture
        intermediate_out = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Capture intermediate output at specified layer
            if self.use_intermediate_ctc and i == self.intermediate_layer:
                intermediate_out = self.intermediate_proj(self.intermediate_ln(x))
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Output projection
        phone_logits = self.fc_out(x)  # [B, num_patches, n_classes + 1]
        
        # Return simple output if no auxiliary losses
        if not self.use_diphone and not self.use_intermediate_ctc:
            return phone_logits
        
        # Build output dictionary with all requested outputs
        outputs = {'phone_logits': phone_logits}
        
        if self.use_diphone:
            outputs['diphone_logits'] = self.diphone_head(x)
        
        if self.use_intermediate_ctc and intermediate_out is not None:
            outputs['intermediate_logits'] = intermediate_out
        
        return outputs

