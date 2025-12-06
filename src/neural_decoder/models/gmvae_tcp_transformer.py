"""
Pre-Patch GMVAE TCP Transformer for Neural Speech Decoding.

Architecture:
    Neural Input [B, T, 256]  (6v only - a44 ignored by augmenter)
           │
           ├──────────────────────────────────────┐
           │                                      ↓
           │                              FrameGMVAE Stream
           │                              [B, T, 256] → [B, T, 40]
           │                                      │
           ↓                                      ↓
      concat ←────────────────────────────────────┘
           │
     [B, T, 296]  (256 neural + 40 cluster probs)
           │
      Patching (stride 5)
           │
     [B, T', 1480]  (296 * 5)
           │
      TCP Transformer
           │
     [B, T', n_classes+1]

Key differences from dual-stream architecture:
- GMVAE sees raw 20ms frames, not 100ms patches
- Cluster probs are extra input channels, not a parallel stream
- No fusion gate - TCP learns to use cluster info naturally
- Simpler loss computation - GMVAE losses at frame level
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from neural_decoder.models.transformer import TransformerBlock


# =============================================================================
# GMVAE Layers (reused from dual_stream_tcp_gmvae.py)
# =============================================================================


class GumbelSoftmax(nn.Module):
    """
    Sample from Gumbel-Softmax distribution with optional hard (one-hot) samples.
    
    Enables differentiable discrete sampling for cluster assignments.
    """
    
    def __init__(self, f_dim, c_dim):
        super().__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim
    
    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)
    
    def gumbel_softmax(self, logits, temperature, hard=False):
        y = self.gumbel_softmax_sample(logits, temperature)
        
        if not hard:
            return y
        
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard
    
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


class Gaussian(nn.Module):
    """Gaussian distribution layer with reparameterization trick."""
    
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z
    
    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


# =============================================================================
# TCP Components (reused from tcp_transformer.py)
# =============================================================================


class MonophoneHead(nn.Module):
    """LN -> Linear head for intermediate CTC supervision (Level 1)."""
    
    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, n_classes + 1)
    
    def forward(self, x):
        return self.proj(self.ln(x))


class DiphoneHead(nn.Module):
    """LN -> Linear head for diphone prediction with marginalization (Level 2)."""
    
    def __init__(self, hidden_dim, n_classes=40, n_diphones=1600):
        super().__init__()
        self.n_classes = n_classes
        self.n_diphones = n_diphones
        self.ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, n_diphones + 1)
        self.raw_logits = None
    
    def forward(self, x):
        logits = self.proj(self.ln(x))
        self.raw_logits = logits
        return self.marginalize(logits)
    
    def marginalize(self, logits):
        B, T, D = logits.shape
        probs = F.softmax(logits, dim=-1)
        diphone_probs = probs[:, :, 1:]
        diphone_probs = diphone_probs.view(B, T, self.n_classes, self.n_classes)
        mono_probs = diphone_probs.sum(dim=2)
        mono_logits = torch.log(mono_probs + 1e-8)
        blank_prob = probs[:, :, 0:1]
        blank_logit_out = torch.log(blank_prob + 1e-8)
        return torch.cat([blank_logit_out, mono_logits], dim=-1)


class ContextHead(nn.Module):
    """Two-layer MLP for context-aware final prediction (Level 3)."""
    
    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes + 1),
        )
    
    def forward(self, x):
        return self.mlp(self.ln(x))


class TCPGatingNetwork(nn.Module):
    """Adaptive weighting of three TCP prediction levels."""
    
    def __init__(self, hidden_dim, use_mask_ratio=True):
        super().__init__()
        self.use_mask_ratio = use_mask_ratio
        input_dim = hidden_dim + 1 if use_mask_ratio else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
    
    def forward(self, context_summary, mask_ratio=0.0):
        B = context_summary.shape[0]
        device = context_summary.device
        
        if self.use_mask_ratio:
            if isinstance(mask_ratio, float):
                mask_ratio_tensor = torch.full((B, 1), mask_ratio, device=device)
            else:
                mask_ratio_tensor = mask_ratio.view(B, 1)
            x = torch.cat([context_summary, mask_ratio_tensor], dim=-1)
        else:
            x = context_summary
        
        logits = self.mlp(x)
        return F.softmax(logits, dim=-1)


class GradientHighway(nn.Module):
    """Residual pathway from early layer to upper layers."""
    
    def __init__(self, hidden_dim, scale=0.1):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = scale
    
    def forward(self, x):
        return self.scale * self.proj(x)


# =============================================================================
# Frame-Level GMVAE (operates on raw 20ms frames before patching)
# =============================================================================


class FrameGMVAE(nn.Module):
    """
    GMVAE operating on raw neural frames (20ms resolution).
    
    This is a lightweight GMVAE that processes individual frames before patching.
    It outputs cluster probabilities that are concatenated to neural features.
    
    Args:
        input_dim: Raw neural channel dimension (256 for 6v only)
        hidden_dim: Internal hidden dimension (smaller than patch GMVAE)
        z_dim: Latent space dimension
        n_clusters: Number of Gaussian mixture components
    """
    
    def __init__(
        self,
        input_dim=256,  # 6v only (a44 ignored by augmenter)
        hidden_dim=256,
        z_dim=64,
        n_clusters=40,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        
        # q(y|x): Frame → cluster assignment
        self.inference_qyx = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.qyx_gumbel = GumbelSoftmax(hidden_dim, n_clusters)
        
        # q(z|x,y): Frame + cluster → latent
        self.inference_qzxy = nn.Sequential(
            nn.Linear(input_dim + n_clusters, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.qzxy_gaussian = Gaussian(hidden_dim, z_dim)
        
        # p(z|y): Prior over z given cluster
        self.pzy_mu = nn.Linear(n_clusters, z_dim)
        self.pzy_var = nn.Linear(n_clusters, z_dim)
        
        # p(x|z): Decoder - reconstruct frames
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def pzy(self, y):
        """Prior p(z|y)."""
        mu = self.pzy_mu(y)
        var = F.softplus(self.pzy_var(y))
        return mu, var
    
    def forward(self, x, temperature=1.0, hard=False):
        """
        Forward pass through Frame GMVAE.
        
        Args:
            x: [B, T, input_dim] raw neural frames (20ms resolution)
            temperature: Gumbel-Softmax temperature
            hard: Whether to use hard (one-hot) cluster assignments
            
        Returns:
            dict with:
                - prob_cat: [B, T, n_clusters] cluster probabilities
                - x_rec: [B, T, input_dim] reconstructed frames
                - z, mu, var: latent variables
                - y: cluster samples
                - logits: raw cluster logits
                - y_mu, y_var: prior parameters
        """
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        
        # q(y|x): Infer cluster from frame
        h_qyx = self.inference_qyx(x_flat)
        logits, prob_cat, y = self.qyx_gumbel(h_qyx, temperature, hard)
        
        # q(z|x,y): Infer latent given frame and cluster
        xy = torch.cat([x_flat, y], dim=-1)
        h_qzxy = self.inference_qzxy(xy)
        mu, var, z = self.qzxy_gaussian(h_qzxy)
        
        # p(z|y): Prior
        y_mu, y_var = self.pzy(y)
        
        # p(x|z): Reconstruct frame
        x_rec = self.decoder(z)
        
        return {
            'prob_cat': prob_cat.reshape(B, T, -1),
            'x_rec': x_rec.reshape(B, T, -1),
            'z': z.reshape(B, T, -1),
            'mu': mu.reshape(B, T, -1),
            'var': var.reshape(B, T, -1),
            'y': y.reshape(B, T, -1),
            'logits': logits.reshape(B, T, -1),
            'y_mu': y_mu.reshape(B, T, -1),
            'y_var': y_var.reshape(B, T, -1),
        }


# =============================================================================
# Pre-Patch GMVAE TCP Transformer
# =============================================================================


class GMVAETCPTransformer(nn.Module):
    """
    Pre-Patch GMVAE TCP Transformer for neural speech decoding.
    
    GMVAE operates on raw neural frames (20ms), cluster probabilities are
    concatenated as extra channels, then the enriched data is patched and
    processed by the TCP Transformer.
    
    Key features:
    - Frame-level GMVAE clustering before patching
    - Cluster probs become extra input channels (40 "virtual electrodes")
    - Full TCP multi-level heads (mono, diphone, context)
    - No fusion gate - TCP learns to use cluster features naturally
    
    Args:
        neural_dim: Raw neural channel dimension (256 for 6v only)
        n_classes: Number of phoneme classes (40)
        hidden_dim: Transformer hidden dimension (896)
        num_layers: Number of Transformer blocks (10)
        num_heads: Number of attention heads (8)
        ffn_dim: Feed-forward network dimension (3584)
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
        gmvae_hidden_dim: GMVAE internal hidden dimension
        gmvae_z_dim: GMVAE latent space dimension
        n_clusters: Number of Gaussian mixture components
        gmvae_init_temp: Initial Gumbel-Softmax temperature
        gmvae_min_temp: Minimum temperature after annealing
        gmvae_decay_temp_rate: Temperature decay rate
        detach_gmvae: Whether to detach cluster probs from gradient
        nDays: Unused, kept for config compatibility
        device: Device to run on
    """
    
    def __init__(
        self,
        neural_dim=256,  # 6v only (256 ch), a44 ignored by augmenter
        n_classes=40,
        hidden_dim=896,
        num_layers=10,
        num_heads=8,
        ffn_dim=3584,
        patch_len=5,
        patch_stride=5,
        dropout=0.35,  # From tcp_xxxl best
        attn_dropout=0.1,
        input_dropout=0.2,
        max_rel_pos=64,
        num_masks=20,  # From tcp_xxxl best
        max_mask_fraction=0.075,  # From tcp_xxxl best
        # TCP parameters
        use_tcp=True,
        mono_layer=3,
        diphone_layer=6,
        use_gating=True,
        use_highway=True,
        fixed_gate_weights=(0.33, 0.33, 0.34),
        n_diphones=1600,
        # Frame GMVAE parameters
        gmvae_hidden_dim=256,
        gmvae_z_dim=64,
        n_clusters=40,
        gmvae_init_temp=1.0,
        gmvae_min_temp=0.5,
        gmvae_decay_temp_rate=0.013,
        detach_gmvae=False,
        # Compatibility
        nDays=24,
        device="cuda",
    ):
        super().__init__()
        
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
        self.n_clusters = n_clusters
        self.detach_gmvae = detach_gmvae
        
        # TCP parameters
        self.use_tcp = use_tcp
        self.mono_layer = mono_layer
        self.diphone_layer = diphone_layer
        self.use_gating = use_gating
        self.use_highway = use_highway
        self.fixed_gate_weights = fixed_gate_weights
        self.n_diphones = n_diphones
        
        # GMVAE temperature scheduling
        self.gmvae_init_temp = gmvae_init_temp
        self.gmvae_min_temp = gmvae_min_temp
        self.gmvae_decay_temp_rate = gmvae_decay_temp_rate
        self.register_buffer('gmvae_temperature', torch.tensor(gmvae_init_temp))
        
        # For CTC length calculation
        self.kernelLen = patch_len
        self.strideLen = patch_stride
        
        # =====================================================================
        # Frame GMVAE (operates on raw 20ms frames before patching)
        # =====================================================================
        
        self.frame_gmvae = FrameGMVAE(
            input_dim=neural_dim,
            hidden_dim=gmvae_hidden_dim,
            z_dim=gmvae_z_dim,
            n_clusters=n_clusters,
        )
        # Normalize and gate cluster features so TCP can smoothly scale their influence
        self.cluster_ln = nn.LayerNorm(n_clusters)
        self.cluster_gate = nn.Parameter(torch.tensor(0.0))  # scalar gate (sigmoid)
        
        # =====================================================================
        # TCP Transformer (takes enriched input: neural + cluster probs)
        # =====================================================================
        
        # Enriched input dimension: neural channels + cluster probs
        enriched_dim = neural_dim + n_clusters  # 256 + clusters
        patch_input_dim = enriched_dim * patch_len  # (256+clusters) * 5
        
        self.patch_embed = nn.Linear(patch_input_dim, hidden_dim)
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
                self.gating = TCPGatingNetwork(hidden_dim, use_mask_ratio=True)
            
            if self.use_highway:
                self.highway = GradientHighway(hidden_dim, scale=0.1)
        
        # Track mask ratio during training
        self._current_mask_ratio = 0.0
    
    def _form_patches(self, x):
        """Form non-overlapping patches from enriched input sequence."""
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
    
    def update_gmvae_temperature(self, epoch):
        """Update GMVAE temperature based on epoch (exponential decay)."""
        new_temp = max(
            self.gmvae_init_temp * math.exp(-self.gmvae_decay_temp_rate * epoch),
            self.gmvae_min_temp
        )
        self.gmvae_temperature.fill_(new_temp)
    
    def forward(self, neuralInput, dayIdx=None):
        """
        Forward pass.
        
        Args:
            neuralInput: [B, T, neural_dim] raw neural features (20ms resolution)
            dayIdx: Unused, kept for API compatibility
            
        Returns:
            dict with:
                - phone_logits: [B, T', n_classes + 1] final output
                - mono_logits: [B, T', n_classes + 1] (if use_tcp)
                - diphone_logits: [B, T', n_diphones + 1] raw (if use_tcp)
                - context_logits: [B, T', n_classes + 1] (if use_tcp)
                - gates: [B, 3] TCP gate values (if use_tcp and use_gating)
                - gmvae_output: dict with GMVAE internals
                - mask_ratio: float
        """
        B, T, D = neuralInput.shape
        
        # =====================================================================
        # 1. Frame GMVAE on raw neural input (20ms resolution)
        # =====================================================================
        
        gmvae_out = self.frame_gmvae(
            neuralInput,
            temperature=self.gmvae_temperature.item(),
            hard=not self.training,
        )
        cluster_probs = gmvae_out['prob_cat']  # [B, T, n_clusters]
        
        # Optionally detach cluster probs to prevent TCP gradients from affecting GMVAE
        if self.detach_gmvae:
            cluster_probs = cluster_probs.detach()
        
        # Normalize + gate cluster features before concatenation
        cluster_norm = self.cluster_ln(cluster_probs)
        cluster_scale = torch.sigmoid(self.cluster_gate)
        cluster_feats = cluster_scale * cluster_norm
        
        # =====================================================================
        # 2. Concatenate cluster probs as extra channels
        # =====================================================================
        
        enriched = torch.cat([neuralInput, cluster_feats], dim=-1)  # [B, T, enriched_dim]
        
        # =====================================================================
        # 3. Form patches from enriched data
        # =====================================================================
        
        patches = self._form_patches(enriched)  # [B, T', 1480]
        
        # =====================================================================
        # 4. TCP Transformer
        # =====================================================================
        
        x = self.patch_embed(patches)
        x = self.input_dropout(x)
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
                'gmvae_output': gmvae_out,
                'mask_ratio': self._current_mask_ratio,
            }
        
        # =====================================================================
        # 5. TCP heads and gated fusion
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
            'gmvae_output': gmvae_out,
            'mask_ratio': self._current_mask_ratio,
        }

