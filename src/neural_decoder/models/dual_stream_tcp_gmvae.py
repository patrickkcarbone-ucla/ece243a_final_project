"""
Dual-Stream TCP + GMVAE Model for Neural Speech Decoding.

Combines:
- Full TCP Stream: Temporal Coarticulation Pyramid Transformer with multi-level heads
  (MonophoneHead, DiphoneHead, ContextHead) and TCP gating
- GMVAE Stream: Gaussian Mixture VAE for clustering/representation learning

The TCP stream produces gated fusion of mono/diphone/context predictions.
The GMVAE stream produces latent representations.
Both streams are fused before the final CTC output.

Architecture:
    Neural Input [B, T, 256]
           │
    ┌──────┴──────┐
    ▼             ▼
 TCP Stream   GMVAE Stream
 (Multi-level)  (Clustering)
    │             │
 TCP Fused    GMVAE Hidden
    │             │
    └──────┬──────┘
           ▼
      Fusion Layer
           │
           ▼
    [B, T', n_classes+1]
         CTC Loss
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from neural_decoder.models.transformer import TransformerBlock


# =============================================================================
# GMVAE Layers (adapted from scripts/GMVAE_UPDATED/networks/Layers.py)
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
# TCP Components (from tcp_transformer.py)
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
# Temporal GMVAE Stream
# =============================================================================


class TemporalGMVAEStream(nn.Module):
    """
    GMVAE adapted for temporal sequences.
    
    Now operates on TCP transformer features instead of raw patches.
    Has dual decoders: one for TCP feature reconstruction, one for raw patch reconstruction.
    
    Args:
        input_dim: Dimension of input features (TCP hidden_dim, e.g., 896)
        patch_dim: Dimension of raw patches for cross-modal reconstruction (e.g., 2560)
        hidden_dim: GMVAE internal hidden dimension
        z_dim: Latent space dimension
        n_clusters: Number of Gaussian mixture components
        output_dim: Output dimension for fusion (defaults to hidden_dim)
    """
    
    def __init__(
        self,
        input_dim,
        patch_dim,
        hidden_dim=768,
        z_dim=128,
        n_clusters=40,
        output_dim=None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.output_dim = output_dim or hidden_dim
        
        # q(y|x): Infer cluster from TCP features
        self.inference_qyx = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.qyx_gumbel = GumbelSoftmax(hidden_dim, n_clusters)
        
        # q(z|x,y): Infer latent given TCP features and cluster
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
        
        # Decoder for TCP features (self-reconstruction)
        self.generative_pxz_tcp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),  # Reconstruct TCP features
        )
        
        # Decoder for raw patches (cross-modal reconstruction)
        self.generative_pxz_patch = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, patch_dim),  # Reconstruct raw patches
            nn.Tanh(),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(z_dim + n_clusters, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def pzy(self, y):
        mu = self.pzy_mu(y)
        var = F.softplus(self.pzy_var(y))
        return mu, var
    
    def forward(self, tcp_features, temperature=1.0, hard=False):
        """
        Forward pass through GMVAE.
        
        Args:
            tcp_features: [B, T', input_dim] TCP transformer intermediate features
            temperature: Gumbel-Softmax temperature
            hard: Whether to use hard (one-hot) cluster assignments
            
        Returns:
            dict with hidden, z, mu, var, y, prob_cat, logits, y_mu, y_var,
            tcp_rec (TCP feature reconstruction), patch_rec (raw patch reconstruction)
        """
        B, T, D = tcp_features.shape
        # Use reshape instead of view for non-contiguous tensors
        x_flat = tcp_features.reshape(B * T, D)
        
        # q(y|x): Infer cluster from TCP features
        h_qyx = self.inference_qyx(x_flat)
        logits, prob_cat, y = self.qyx_gumbel(h_qyx, temperature, hard)
        
        # q(z|x,y): Infer latent given TCP features and cluster
        xy = torch.cat([x_flat, y], dim=-1)
        h_qzxy = self.inference_qzxy(xy)
        mu, var, z = self.qzxy_gaussian(h_qzxy)
        
        # p(z|y): Prior
        y_mu, y_var = self.pzy(y)
        
        # Decode to both targets
        tcp_rec = self.generative_pxz_tcp(z)
        patch_rec = self.generative_pxz_patch(z)
        
        # Output for fusion
        zy = torch.cat([z, y], dim=-1)
        hidden = self.output_proj(zy)
        
        return {
            'hidden': hidden.reshape(B, T, -1),
            'z': z.reshape(B, T, -1),
            'mu': mu.reshape(B, T, -1),
            'var': var.reshape(B, T, -1),
            'y': y.reshape(B, T, -1),
            'prob_cat': prob_cat.reshape(B, T, -1),
            'logits': logits.reshape(B, T, -1),
            'y_mu': y_mu.reshape(B, T, -1),
            'y_var': y_var.reshape(B, T, -1),
            'tcp_rec': tcp_rec.reshape(B, T, -1),
            'patch_rec': patch_rec.reshape(B, T, -1),
        }


# =============================================================================
# Dual-Stream TCP + GMVAE Model
# =============================================================================


class DualStreamTCPGMVAE(nn.Module):
    """
    Dual-stream model combining full TCP Transformer and GMVAE.
    
    TCP Stream features:
    - MonophoneHead at mono_layer
    - DiphoneHead at diphone_layer  
    - ContextHead at final layer
    - TCP gating network for adaptive level weighting
    - Gradient highway for stable training
    
    GMVAE Stream features:
    - Temporal patch-wise clustering
    - Latent space learning
    - Reconstruction for auxiliary loss
    
    Fusion:
    - TCP produces gated fusion of mono/diphone/context
    - GMVAE produces hidden representation
    - Both are combined via learned fusion
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
        gmvae_hidden_dim=768,
        gmvae_z_dim=128,
        n_clusters=40,
        gmvae_init_temp=1.0,
        gmvae_min_temp=0.5,
        gmvae_decay_temp_rate=0.013,
        gmvae_tap_layer=3,  # Which TCP layer feeds GMVAE (default: mono_layer)
        detach_gmvae_input=False,  # Whether to detach TCP features from gradient
        # Fusion parameters
        fusion_method="gated",
        fusion_dropout=0.1,
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
        self.fusion_method = fusion_method
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
        
        # GMVAE temperature scheduling
        self.gmvae_init_temp = gmvae_init_temp
        self.gmvae_min_temp = gmvae_min_temp
        self.gmvae_decay_temp_rate = gmvae_decay_temp_rate
        self.register_buffer('gmvae_temperature', torch.tensor(gmvae_init_temp))
        
        # GMVAE TCP tap configuration
        self.gmvae_tap_layer = gmvae_tap_layer
        self.detach_gmvae_input = detach_gmvae_input
        
        # For CTC length calculation
        self.kernelLen = patch_len
        self.strideLen = patch_stride
        
        patch_input_dim = neural_dim * patch_len
        
        # =====================================================================
        # TCP Stream
        # =====================================================================
        
        self.tcp_patch_embed = nn.Linear(patch_input_dim, hidden_dim)
        self.tcp_input_dropout = nn.Dropout(input_dropout)
        
        self.tcp_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.tcp_mask_token, std=0.02)
        
        self.tcp_blocks = nn.ModuleList([
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
        
        self.tcp_ln = nn.LayerNorm(hidden_dim)
        
        # TCP heads
        if self.use_tcp:
            self.mono_head = MonophoneHead(hidden_dim, n_classes)
            self.diphone_head = DiphoneHead(hidden_dim, n_classes, n_diphones)
            self.context_head = ContextHead(hidden_dim, n_classes)
            
            if self.use_gating:
                self.tcp_gating = TCPGatingNetwork(hidden_dim, use_mask_ratio=True)
            
            if self.use_highway:
                self.highway = GradientHighway(hidden_dim, scale=0.1)
        
        # Track mask ratio for TCP gating
        self._current_mask_ratio = 0.0
        
        # =====================================================================
        # GMVAE Stream (operates on TCP features, not raw patches)
        # =====================================================================
        
        self.gmvae_stream = TemporalGMVAEStream(
            input_dim=hidden_dim,      # TCP features (896)
            patch_dim=patch_input_dim, # Raw patches (2560) for cross-modal reconstruction
            hidden_dim=gmvae_hidden_dim,
            z_dim=gmvae_z_dim,
            n_clusters=n_clusters,
            output_dim=hidden_dim,
        )
        
        # =====================================================================
        # Final Fusion (TCP fused output + GMVAE hidden)
        # =====================================================================
        
        # Project TCP logits to hidden for fusion
        self.tcp_to_hidden = nn.Linear(n_classes + 1, hidden_dim)
        
        if fusion_method == "gated":
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        elif fusion_method == "concat":
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion_method == "attention":
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=fusion_dropout,
                batch_first=True,
            )
        
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.fusion_ln = nn.LayerNorm(hidden_dim)
        
        # Final output
        self.fc_out = nn.Linear(hidden_dim, n_classes + 1)
    
    def _form_patches(self, x):
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
        x = torch.where(mask_expanded, x, self.tcp_mask_token.expand_as(x))
        return x
    
    def _apply_tcp_gates(self, z_mono, z_diphone, z_context, gates):
        g_mono = gates[:, 0:1].unsqueeze(-1)
        g_diphone = gates[:, 1:2].unsqueeze(-1)
        g_context = gates[:, 2:3].unsqueeze(-1)
        return g_mono * z_mono + g_diphone * z_diphone + g_context * z_context
    
    def _apply_fixed_tcp_gates(self, z_mono, z_diphone, z_context):
        w_mono, w_diphone, w_context = self.fixed_gate_weights
        return w_mono * z_mono + w_diphone * z_diphone + w_context * z_context
    
    def _fuse_tcp_gmvae(self, h_tcp, h_gmvae):
        """Fuse TCP logits (projected to hidden) with GMVAE hidden."""
        gate = None
        
        if self.fusion_method == "gated":
            concat = torch.cat([h_tcp, h_gmvae], dim=-1)
            gate = self.fusion_gate(concat)
            h_fused = gate * h_tcp + (1 - gate) * h_gmvae
        elif self.fusion_method == "concat":
            concat = torch.cat([h_tcp, h_gmvae], dim=-1)
            h_fused = self.fusion_proj(concat)
        elif self.fusion_method == "add":
            h_fused = h_tcp + h_gmvae
        elif self.fusion_method == "attention":
            h_fused, _ = self.fusion_attn(query=h_tcp, key=h_gmvae, value=h_gmvae)
            h_fused = h_fused + h_tcp
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
        
        h_fused = self.fusion_dropout(h_fused)
        h_fused = self.fusion_ln(h_fused)
        return h_fused, gate
    
    def update_gmvae_temperature(self, epoch):
        new_temp = max(
            self.gmvae_init_temp * math.exp(-self.gmvae_decay_temp_rate * epoch),
            self.gmvae_min_temp
        )
        self.gmvae_temperature.fill_(new_temp)
    
    def forward(self, neuralInput, dayIdx=None):
        """
        Forward pass.
        
        The GMVAE now operates on TCP transformer features (tapped at gmvae_tap_layer)
        instead of raw patches. This provides phonetically-structured input for clustering.
        
        Returns:
            dict with:
                - phone_logits: [B, T', n_classes + 1] final fused output
                - mono_logits: [B, T', n_classes + 1] (if use_tcp)
                - diphone_logits: [B, T', n_diphones + 1] raw (if use_tcp)
                - context_logits: [B, T', n_classes + 1] (if use_tcp)
                - tcp_fused_logits: [B, T', n_classes + 1] TCP gated fusion (if use_tcp)
                - tcp_gates: [B, 3] TCP gate values (if use_tcp and use_gating)
                - gmvae_output: dict with GMVAE internals (includes tcp_rec, patch_rec)
                - fusion_gate: [B, T', 1] final fusion gate (if gated fusion)
                - patches: [B, T', patch_dim] raw input patches (for patch_rec loss)
                - gmvae_tcp_input: [B, T', hidden_dim] TCP features fed to GMVAE (for tcp_rec loss)
                - mask_ratio: float
        """
        # Form patches
        patches = self._form_patches(neuralInput)
        B, T, _ = patches.shape
        
        # =====================================================================
        # TCP Stream (runs first to generate features for GMVAE)
        # =====================================================================
        
        x = self.tcp_patch_embed(patches)
        x = self.tcp_input_dropout(x)
        x = self._apply_time_masking(x)
        
        # Process through Transformer blocks with TCP tap points
        intermediate_outputs = {}
        highway_input = None
        gmvae_input = None  # Will capture at gmvae_tap_layer
        
        for i, block in enumerate(self.tcp_blocks):
            if self.use_tcp and self.use_highway and highway_input is not None:
                if i >= self.mono_layer + 2:
                    x = x + self.highway(highway_input)
            
            x = block(x)
            
            if self.use_tcp:
                if i == self.mono_layer:
                    intermediate_outputs["mono_hidden"] = x
                    if self.use_highway:
                        highway_input = x.detach()
                if i == self.diphone_layer:
                    intermediate_outputs["diphone_hidden"] = x
            
            # Capture GMVAE input at configured tap layer
            if i == self.gmvae_tap_layer:
                gmvae_input = x.detach() if self.detach_gmvae_input else x
        
        x = self.tcp_ln(x)
        
        # Fallback: if gmvae_tap_layer > num_layers, use final output
        if gmvae_input is None:
            gmvae_input = x.detach() if self.detach_gmvae_input else x
        
        # TCP head outputs
        if self.use_tcp:
            z_mono = self.mono_head(intermediate_outputs["mono_hidden"])
            z_diphone_marginalized = self.diphone_head(intermediate_outputs["diphone_hidden"])
            z_context = self.context_head(x)
            
            # TCP gated fusion
            if self.use_gating:
                context_summary = x.mean(dim=1)
                tcp_gates = self.tcp_gating(context_summary, self._current_mask_ratio)
                tcp_fused = self._apply_tcp_gates(z_mono, z_diphone_marginalized, z_context, tcp_gates)
            else:
                tcp_gates = None
                tcp_fused = self._apply_fixed_tcp_gates(z_mono, z_diphone_marginalized, z_context)
        else:
            # Simple output if not using TCP
            tcp_fused = self.fc_out(x)
            z_mono = z_diphone_marginalized = z_context = None
            tcp_gates = None
        
        # =====================================================================
        # GMVAE Stream (operates on TCP features, not raw patches)
        # =====================================================================
        
        gmvae_out = self.gmvae_stream(
            gmvae_input,  # TCP features instead of raw patches
            temperature=self.gmvae_temperature.item(),
            hard=not self.training,
        )
        h_gmvae = gmvae_out['hidden']
        
        # =====================================================================
        # Final Fusion: TCP fused logits + GMVAE hidden
        # =====================================================================
        
        # Project TCP logits to hidden space for fusion
        h_tcp = self.tcp_to_hidden(tcp_fused)
        
        # Fuse TCP and GMVAE
        h_fused, fusion_gate = self._fuse_tcp_gmvae(h_tcp, h_gmvae)
        
        # Final output
        phone_logits = self.fc_out(h_fused)
        
        # Build output dict
        output = {
            'phone_logits': phone_logits,
            'gmvae_output': gmvae_out,
            'fusion_gate': fusion_gate,
            'patches': patches,  # Raw patches for patch reconstruction loss
            'gmvae_tcp_input': gmvae_input,  # TCP features for TCP reconstruction loss
            'mask_ratio': self._current_mask_ratio,
        }
        
        if self.use_tcp:
            output['mono_logits'] = z_mono
            output['diphone_logits'] = self.diphone_head.raw_logits
            output['context_logits'] = z_context
            output['tcp_fused_logits'] = tcp_fused
            output['tcp_gates'] = tcp_gates
        
        return output
