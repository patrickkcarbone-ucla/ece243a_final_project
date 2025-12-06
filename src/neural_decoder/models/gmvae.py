"""
Standalone GMVAE for Neural Speech Decoding.

This module implements a Gaussian Mixture Variational Autoencoder (GMVAE)
that operates on 100ms neural patches. It can be trained independently
and later used as a feature extractor for TCP models.

Architecture:
    Neural Input [B, T, 256] (20ms resolution)
           |
       Patching (5 bins = 100ms)
           |
    [B, T', 1280] (256 * 5 flattened)
           |
       GMVAE Encoder
           |
       q(y|x): [B, T', K] cluster probs (Gumbel-Softmax)
       q(z|x,y): [B, T', z_dim] latent (Gaussian)
           |
       GMVAE Decoder
           |
    [B, T', 1280] reconstructed patches

The model also includes an evaluation head for computing PER
(Phoneme Error Rate) without affecting GMVAE training.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


# =============================================================================
# Core GMVAE Components
# =============================================================================


class GumbelSoftmax(nn.Module):
    """
    Gumbel-Softmax distribution for differentiable discrete sampling.

    Enables gradient flow through categorical cluster assignments
    using the Gumbel-Softmax reparameterization trick.

    Args:
        f_dim: Input feature dimension
        c_dim: Number of categories (clusters)
    """

    def __init__(self, f_dim, c_dim):
        super().__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        """Sample from Gumbel(0, 1) distribution."""
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        """Sample from Gumbel-Softmax distribution."""
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        Gumbel-Softmax with optional straight-through estimator.

        Args:
            logits: Raw logits for each category
            temperature: Softmax temperature (lower = sharper)
            hard: If True, use straight-through estimator for hard samples
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        # Straight-through estimator: hard forward, soft backward
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False):
        """
        Forward pass.

        Args:
            x: [*, f_dim] input features
            temperature: Gumbel-Softmax temperature
            hard: Whether to use hard (one-hot) samples

        Returns:
            logits: [*, c_dim] raw logits
            prob: [*, c_dim] softmax probabilities
            y: [*, c_dim] Gumbel-Softmax samples
        """
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


class Gaussian(nn.Module):
    """
    Gaussian distribution layer with reparameterization trick.

    Args:
        in_dim: Input dimension
        z_dim: Latent space dimension
    """

    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, var):
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [*, in_dim] input features

        Returns:
            mu: [*, z_dim] mean
            var: [*, z_dim] variance (softplus activated)
            z: [*, z_dim] reparameterized sample
        """
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


class PhonemeEvalHead(nn.Module):
    """
    Evaluation head for computing PER from cluster probabilities.

    This head maps cluster probabilities to phoneme logits for CTC decoding.
    It is trained separately and gradients do NOT flow back to GMVAE.

    This allows fair evaluation of both:
    - Unsupervised GMVAE: learns optimal cluster→phoneme mapping
    - Supervised GMVAE: learns near-identity mapping

    Args:
        n_clusters: Number of GMVAE clusters
        n_phonemes: Number of phoneme classes (default 40)
    """

    def __init__(self, n_clusters, n_phonemes=40):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_phonemes = n_phonemes
        # +1 for CTC blank token
        self.proj = nn.Linear(n_clusters, n_phonemes + 1)

    def forward(self, cluster_probs):
        """
        Map cluster probabilities to phoneme logits.

        Args:
            cluster_probs: [B, T', n_clusters] cluster probabilities

        Returns:
            phoneme_logits: [B, T', n_phonemes + 1] logits for CTC
        """
        return self.proj(cluster_probs)


# =============================================================================
# Standalone GMVAE Model (100ms patches)
# =============================================================================


class StandaloneGMVAE(nn.Module):
    """
    Standalone GMVAE operating on 100ms neural patches.

    This model processes raw neural data by:
    1. Forming non-overlapping 100ms patches (5 x 20ms bins)
    2. Encoding patches to cluster assignments via Gumbel-Softmax
    3. Encoding patches + clusters to Gaussian latent space
    4. Decoding latent back to reconstructed patches

    The model includes an evaluation head for PER monitoring that
    does not affect GMVAE training.

    Args:
        neural_dim: Raw neural channel dimension (256 for 6v)
        patch_len: Number of time bins per patch (5 = 100ms)
        patch_stride: Stride between patches (5 = non-overlapping)
        hidden_dim: GMVAE internal hidden dimension
        z_dim: Latent space dimension
        n_clusters: Number of Gaussian mixture components
        n_phonemes: Number of phoneme classes for eval head
        init_temp: Initial Gumbel-Softmax temperature
        min_temp: Minimum temperature after annealing
        decay_temp_rate: Temperature decay rate per epoch
    """

    def __init__(
        self,
        neural_dim=256,
        patch_len=5,
        patch_stride=5,
        hidden_dim=512,
        z_dim=64,
        n_clusters=40,
        n_phonemes=40,
        init_temp=2.0,
        min_temp=0.5,
        decay_temp_rate=0.013,
        nDays=24,  # Unused, for config compatibility
        device="cuda",
    ):
        super().__init__()

        # Store parameters
        self.neural_dim = neural_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.n_phonemes = n_phonemes
        self.device = device

        # Patch dimension after flattening
        self.patch_dim = neural_dim * patch_len  # 256 * 5 = 1280

        # For CTC length calculation (compatible with trainer)
        self.kernelLen = patch_len
        self.strideLen = patch_stride

        # Temperature scheduling
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.decay_temp_rate = decay_temp_rate
        self.register_buffer('temperature', torch.tensor(init_temp))

        # =====================================================================
        # GMVAE Encoder: q(y|x) - Infer cluster from patch
        # =====================================================================

        self.inference_qyx = nn.Sequential(
            nn.Linear(self.patch_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
        )
        self.qyx_gumbel = GumbelSoftmax(hidden_dim, n_clusters)

        # =====================================================================
        # GMVAE Encoder: q(z|x,y) - Infer latent given patch and cluster
        # =====================================================================

        self.inference_qzxy = nn.Sequential(
            nn.Linear(self.patch_dim + n_clusters, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
        )
        self.qzxy_gaussian = Gaussian(hidden_dim, z_dim)

        # =====================================================================
        # GMVAE Prior: p(z|y) - Prior over z given cluster
        # =====================================================================

        self.pzy_mu = nn.Linear(n_clusters, z_dim)
        self.pzy_var = nn.Linear(n_clusters, z_dim)

        # =====================================================================
        # GMVAE Decoder: p(x|z) - Reconstruct patch from latent
        # =====================================================================

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.patch_dim),
        )

        # =====================================================================
        # Evaluation Head (for PER monitoring, not used in GMVAE loss)
        # =====================================================================

        self.eval_head = PhonemeEvalHead(n_clusters, n_phonemes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _form_patches(self, x):
        """
        Form non-overlapping patches from input sequence.

        Args:
            x: [B, T, neural_dim] raw neural input (20ms resolution)

        Returns:
            patches: [B, T', patch_dim] flattened patches (100ms)
        """
        B, T, D = x.shape

        # Calculate number of complete patches
        num_patches = (T - self.patch_len) // self.patch_stride + 1
        valid_len = (num_patches - 1) * self.patch_stride + self.patch_len

        # Trim to valid length
        x = x[:, :valid_len, :]

        # Reshape to patches: [B, T, D] -> [B, num_patches, patch_len, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = x.unfold(2, self.patch_len, self.patch_stride)  # [B, D, num_patches, patch_len]
        x = x.permute(0, 2, 3, 1)  # [B, num_patches, patch_len, D]
        x = x.reshape(B, num_patches, -1)  # [B, num_patches, patch_dim]

        return x

    def pzy(self, y):
        """
        Prior distribution p(z|y).

        Args:
            y: [*, n_clusters] cluster assignment (one-hot or soft)

        Returns:
            mu: [*, z_dim] prior mean
            var: [*, z_dim] prior variance
        """
        mu = self.pzy_mu(y)
        var = F.softplus(self.pzy_var(y))
        return mu, var

    def update_temperature(self, epoch):
        """
        Update Gumbel-Softmax temperature based on epoch.

        Uses exponential decay: temp = init_temp * exp(-decay_rate * epoch)
        """
        new_temp = max(
            self.init_temp * math.exp(-self.decay_temp_rate * epoch),
            self.min_temp
        )
        self.temperature.fill_(new_temp)

    def encode(self, patches, temperature=None, hard=False):
        """
        Encode patches to cluster assignments and latent variables.

        Args:
            patches: [B, T', patch_dim] flattened patches
            temperature: Gumbel-Softmax temperature (uses self.temperature if None)
            hard: Whether to use hard (one-hot) cluster assignments

        Returns:
            dict with cluster and latent information
        """
        if temperature is None:
            temperature = self.temperature.item()

        B, T, D = patches.shape
        patches_flat = patches.reshape(B * T, D)

        # q(y|x): Infer cluster from patch
        h_qyx = self.inference_qyx(patches_flat)
        logits, prob_cat, y = self.qyx_gumbel(h_qyx, temperature, hard)

        # q(z|x,y): Infer latent given patch and cluster
        xy = torch.cat([patches_flat, y], dim=-1)
        h_qzxy = self.inference_qzxy(xy)
        mu, var, z = self.qzxy_gaussian(h_qzxy)

        # p(z|y): Prior
        y_mu, y_var = self.pzy(y)

        return {
            'prob_cat': prob_cat.reshape(B, T, -1),
            'y': y.reshape(B, T, -1),
            'logits': logits.reshape(B, T, -1),
            'z': z.reshape(B, T, -1),
            'mu': mu.reshape(B, T, -1),
            'var': var.reshape(B, T, -1),
            'y_mu': y_mu.reshape(B, T, -1),
            'y_var': y_var.reshape(B, T, -1),
        }

    def decode(self, z):
        """
        Decode latent variables to reconstructed patches.

        Args:
            z: [B, T', z_dim] latent variables

        Returns:
            x_rec: [B, T', patch_dim] reconstructed patches
        """
        B, T, D = z.shape
        z_flat = z.reshape(B * T, D)
        x_rec_flat = self.decoder(z_flat)
        return x_rec_flat.reshape(B, T, -1)

    def forward(self, neural_input, dayIdx=None):
        """
        Full forward pass.

        Args:
            neural_input: [B, T, neural_dim] raw neural features (20ms resolution)
            dayIdx: Unused, kept for API compatibility

        Returns:
            dict with:
                - patches: [B, T', patch_dim] input patches
                - x_rec: [B, T', patch_dim] reconstructed patches
                - prob_cat: [B, T', n_clusters] cluster probabilities
                - logits: [B, T', n_clusters] cluster logits
                - y: [B, T', n_clusters] cluster samples
                - z: [B, T', z_dim] latent samples
                - mu: [B, T', z_dim] posterior mean
                - var: [B, T', z_dim] posterior variance
                - y_mu: [B, T', z_dim] prior mean
                - y_var: [B, T', z_dim] prior variance
                - phone_logits: [B, T', n_phonemes+1] eval head output
        """
        # Form 100ms patches
        patches = self._form_patches(neural_input)

        # Encode
        enc_out = self.encode(
            patches,
            temperature=self.temperature.item(),
            hard=not self.training,
        )

        # Decode
        x_rec = self.decode(enc_out['z'])

        # Evaluation head (detached for PER computation)
        # Note: During training, trainer will handle detachment
        phone_logits = self.eval_head(enc_out['prob_cat'])

        return {
            'patches': patches,
            'x_rec': x_rec,
            'prob_cat': enc_out['prob_cat'],
            'logits': enc_out['logits'],
            'y': enc_out['y'],
            'z': enc_out['z'],
            'mu': enc_out['mu'],
            'var': enc_out['var'],
            'y_mu': enc_out['y_mu'],
            'y_var': enc_out['y_var'],
            'phone_logits': phone_logits,
        }

    def get_cluster_features(self, neural_input):
        """
        Get cluster probabilities for use as features (Phase 2).

        This method is used when GMVAE is frozen and serves as
        a feature extractor for TCP.

        Args:
            neural_input: [B, T, neural_dim] raw neural features

        Returns:
            cluster_probs: [B, T', n_clusters] cluster probabilities
        """
        with torch.no_grad():
            patches = self._form_patches(neural_input)
            enc_out = self.encode(patches, hard=True)
            return enc_out['prob_cat']