"""
GMVAE Loss Functions for Dual-Stream TCP + GMVAE model.

Adapted from scripts/GMVAE_UPDATED/losses/LossFunctions.py
for use with temporal sequence data.

Loss functions:
- reconstruction_loss: MSE/BCE between input patches and reconstruction
- gaussian_kl_loss: KL divergence between q(z|x,y) and p(z|y)
- categorical_entropy_loss: Entropy regularization for cluster assignments
"""

import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class GMVAELossFunctions:
    """
    Collection of loss functions for GMVAE training.

    Adapted from the original GMVAE implementation with modifications
    for temporal sequence processing.
    """

    eps = 1e-8

    def reconstruction_loss(self, real, predicted, rec_type="mse"):
        """
        Reconstruction loss between input patches and reconstructions.

        Args:
            real: [B, T', D] or [B*T', D] input patches
            predicted: [B, T', D] or [B*T', D] reconstructed patches
            rec_type: 'mse' for mean squared error, 'bce' for binary cross entropy

        Returns:
            Scalar loss value
        """
        if rec_type == "mse":
            loss = (real - predicted).pow(2)
        elif rec_type == "bce":
            loss = F.binary_cross_entropy(predicted, real, reduction="none")
        else:
            raise ValueError(
                f"Invalid rec_type '{rec_type}'. Use 'mse' or 'bce'."
            )

        # Mean over all dimensions (normalized by feature count)
        return loss.mean()

    def log_normal(self, x, mu, var):
        """
        Log probability of x under Gaussian with mean=mu and variance=var.

        log N(x|μ, σ²) = -0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)

        Args:
            x: [*, D] samples
            mu: [*, D] means
            var: [*, D] variances

        Returns:
            [*] log probabilities (summed over D)
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var,
            dim=-1,
        )

    def gaussian_kl_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        """
        KL divergence between inference distribution q(z|x,y) and prior p(z|y).

        KL(q||p) = E_q[log q(z|x,y) - log p(z|y)]

        Args:
            z: [B, T', D] or [B*T', D] sampled latent variables
            z_mu: [B, T', D] mean of q(z|x,y)
            z_var: [B, T', D] variance of q(z|x,y)
            z_mu_prior: [B, T', D] mean of p(z|y)
            z_var_prior: [B, T', D] variance of p(z|y)

        Returns:
            Scalar KL divergence loss
        """
        # log q(z|x,y) - log p(z|y)
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(
            z, z_mu_prior, z_var_prior
        )
        return loss.mean()

    def categorical_entropy_loss(self, logits, prob_cat):
        """
        Negative entropy of categorical distribution (encourages confident assignments).

        The original GMVAE uses: -H(q(y|x)) - log(1/K) = E_q[log q(y|x)] + log(K)
        This encourages the model to make confident cluster assignments while
        maintaining diversity across the dataset.

        Args:
            logits: [B, T', K] raw logits for cluster assignment
            prob_cat: [B, T', K] softmax probabilities q(y|x)

        Returns:
            Scalar entropy loss (negative entropy)
        """
        log_q = F.log_softmax(logits, dim=-1)
        # Negative entropy: E_q[log q]
        neg_entropy = torch.sum(prob_cat * log_q, dim=-1)
        return (
            -neg_entropy.mean()
        )  # Return positive value (minimize to maximize entropy)

    def cluster_diversity_loss(self, prob_cat):
        """
        Encourage uniform cluster usage across the batch.

        Maximizes entropy of the marginal distribution p(y) = E_x[q(y|x)].
        This prevents cluster collapse where all inputs map to one cluster.

        Args:
            prob_cat: [B, T', K] softmax probabilities q(y|x)

        Returns:
            Scalar diversity loss (negative entropy of marginal)
        """
        # Average probability per cluster across batch and time
        avg_prob = prob_cat.mean(dim=[0, 1])  # [K]
        # Maximize entropy of marginal (uniform usage)
        entropy = -torch.sum(avg_prob * torch.log(avg_prob + self.eps))
        return -entropy  # Return negative so minimizing increases entropy

    def supervised_cluster_loss(
        self, prob_cat, phone_labels, output_lens, label_lens
    ):
        """
        Supervised clustering loss: encourage cluster assignments to match phoneme labels.

        Uses soft cross-entropy between cluster probabilities and phoneme labels.
        Only applies to frames that can be aligned (uses uniform distribution over
        label phonemes for each frame).

        Args:
            prob_cat: [B, T', K] softmax probabilities q(y|x)
            phone_labels: [B, L] phoneme label sequences (1-indexed, 0 is blank)
            output_lens: [B] number of valid output frames
            label_lens: [B] number of valid label phonemes

        Returns:
            Scalar supervised cluster loss
        """
        B, T, K = prob_cat.shape
        device = prob_cat.device

        total_loss = 0.0
        n_valid = 0

        for b in range(B):
            t_len = min(output_lens[b].item(), T)
            l_len = label_lens[b].item()

            if t_len <= 0 or l_len <= 0:
                continue

            # Get the phonemes for this sample (convert from 1-indexed to 0-indexed for cluster targets)
            # Phone labels are 1-indexed (0 is blank), but clusters are 0-indexed
            phones = phone_labels[b, :l_len]  # [L]

            # Simple frame-level supervision: distribute labels uniformly across frames
            # Each frame gets supervised toward one of the phonemes
            frame_targets = torch.zeros(t_len, K, device=device)

            for t in range(t_len):
                # Map frame to label position (simple linear mapping)
                label_idx = min(int(t * l_len / t_len), l_len - 1)
                phone_id = phones[label_idx].item()

                # Phone IDs are 1-indexed, clusters are 0-indexed
                # Map phone_id (1-40) to cluster (0-39)
                if 1 <= phone_id <= K:
                    cluster_idx = phone_id - 1  # Convert to 0-indexed
                    frame_targets[t, cluster_idx] = 1.0

            # Cross-entropy between cluster probs and targets
            # log(prob_cat[b, :t_len]) * frame_targets -> sum over K, mean over T
            log_probs = torch.log(prob_cat[b, :t_len] + self.eps)  # [T', K]
            frame_loss = -torch.sum(frame_targets * log_probs, dim=-1)  # [T']
            total_loss += frame_loss.mean()
            n_valid += 1

        if n_valid > 0:
            return total_loss / n_valid
        else:
            return torch.tensor(0.0, device=device)


# Standalone function versions for convenience


def reconstruction_loss(real, predicted, rec_type="mse"):
    """
    Reconstruction loss between input patches and reconstructions.

    Args:
        real: Input patches
        predicted: Reconstructed patches
        rec_type: 'mse' or 'bce'

    Returns:
        Scalar loss
    """
    return GMVAELossFunctions().reconstruction_loss(real, predicted, rec_type)


def gaussian_kl_loss(z, z_mu, z_var, z_mu_prior, z_var_prior):
    """
    KL divergence between q(z|x,y) and p(z|y).

    Args:
        z: Sampled latent variables
        z_mu, z_var: Inference distribution parameters
        z_mu_prior, z_var_prior: Prior distribution parameters

    Returns:
        Scalar KL loss
    """
    return GMVAELossFunctions().gaussian_kl_loss(
        z, z_mu, z_var, z_mu_prior, z_var_prior
    )


def categorical_entropy_loss(logits, prob_cat):
    """
    Negative entropy of categorical distribution.

    Args:
        logits: Raw cluster logits
        prob_cat: Softmax probabilities

    Returns:
        Scalar entropy loss
    """
    return GMVAELossFunctions().categorical_entropy_loss(logits, prob_cat)
