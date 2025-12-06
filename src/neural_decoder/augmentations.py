import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise


class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(
                    dim
                )
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(
            input, weight=self.weight, groups=self.groups, padding="same"
        )


class AugmentationPipeline(nn.Module):
    """Standard augmentation pipeline for single-region data."""

    def __init__(self, white_noise_sd=0.0, constant_offset_sd=0.0):
        super().__init__()
        self.white_noise_sd = white_noise_sd
        self.constant_offset_sd = constant_offset_sd

    def forward(self, x, x_44=None):
        """
        Args:
            x: [B, T, C] neural features (6v)
            x_44: ignored for backward compatibility
        """
        if self.training:
            if self.white_noise_sd > 0:
                x = x + torch.randn_like(x) * self.white_noise_sd
            if self.constant_offset_sd > 0:
                x = x + (
                    torch.randn([x.shape[0], 1, x.shape[2]], device=x.device)
                    * self.constant_offset_sd
                )
        return x


# =============================================================================
# Area 44-Based Augmentation Strategies
# =============================================================================


class CrossAreaMixup(nn.Module):
    """
    Mix Area 6v with its temporally-aligned Area 44 signal from the same trial.

    This teaches the model to ignore patterns that look like Area 44 (noise),
    forcing it to learn features unique to Area 6v (speech signal).

    Args:
        mix_ratio_max: Maximum mixing ratio (0 = no mixing, 1 = full replacement)
        shuffle_44: If True, shuffle 44 across time before mixing (control experiment)
    """

    def __init__(self, mix_ratio_max=0.2, shuffle_44=False):
        super().__init__()
        self.mix_ratio_max = mix_ratio_max
        self.shuffle_44 = shuffle_44

    def forward(self, x_6v, x_44):
        """
        Args:
            x_6v: [B, T, 256] Area 6v features
            x_44: [B, T, 256] Area 44 features (temporally aligned)
        Returns:
            x_aug: [B, T, 256] Augmented 6v features
        """
        if self.training and self.mix_ratio_max > 0:
            # Optionally shuffle 44 to destroy temporal alignment (control)
            if self.shuffle_44:
                # Shuffle time dimension independently for each sample
                B, T, C = x_44.shape
                perm = torch.randperm(T, device=x_44.device)
                x_44 = x_44[:, perm, :]

            # Random mix ratio per sample in batch
            lam = (
                torch.rand(x_6v.shape[0], 1, 1, device=x_6v.device)
                * self.mix_ratio_max
            )
            x_aug = (1 - lam) * x_6v + lam * x_44
            return x_aug
        return x_6v


class AdaptiveNoise44(nn.Module):
    """
    Add noise to Area 6v scaled by Area 44's instantaneous variance.

    Intuition: When Area 44 is highly variable, the brain state may be "noisier",
    so we add more noise to 6v. When 44 is stable, we add less noise.

    Args:
        base_noise: Base noise standard deviation (always applied)
        adaptive_scale: How much 44's variance modulates the noise
    """

    def __init__(self, base_noise=0.4, adaptive_scale=0.3):
        super().__init__()
        self.base_noise = base_noise
        self.adaptive_scale = adaptive_scale

    def forward(self, x_6v, x_44):
        """
        Args:
            x_6v: [B, T, 256] Area 6v features
            x_44: [B, T, 256] Area 44 features
        Returns:
            x_aug: [B, T, 256] Augmented 6v features with state-dependent noise
        """
        if self.training and (self.base_noise > 0 or self.adaptive_scale > 0):
            # Compute variance across 44 channels at each timestep
            var_44 = x_44.var(dim=-1, keepdim=True)  # [B, T, 1]

            # Normalize variance to roughly [0, 1] range using sigmoid
            var_norm = (var_44 - var_44.mean()) / (var_44.std() + 1e-8)
            var_norm = torch.sigmoid(var_norm)  # Bound to [0, 1]

            # Scale noise by 44's variance
            noise_scale = self.base_noise + self.adaptive_scale * var_norm
            noise = torch.randn_like(x_6v) * noise_scale

            return x_6v + noise
        return x_6v


class CorrelationGuidedDropout(nn.Module):
    """
    Dropout Area 6v channels that correlate highly with Area 44.

    Channels in 6v that correlate with 44 are likely dominated by shared noise
    rather than speech-specific signal. This augmentation preferentially drops
    those channels to force the model to rely on cleaner channels.

    Args:
        dropout_prob: Probability of dropping correlated channels
        correlation_threshold: Channels with correlation > this are considered "noisy"
        correlation_file: Path to precomputed correlation matrix (.npy file)
    """

    def __init__(
        self,
        dropout_prob=0.3,
        correlation_threshold=0.3,
        correlation_file=None,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.correlation_threshold = correlation_threshold

        # Will be set by load_correlations() or register_noisy_channels()
        self.register_buffer("noisy_channel_mask", None)

        if correlation_file is not None:
            self.load_correlations(correlation_file)

    def load_correlations(self, filepath):
        """Load precomputed 6v-44 correlations and create noisy channel mask."""
        import numpy as np

        # Load correlation matrix [256 x 256] - correlations between 6v and 44 channels
        corr_matrix = np.load(filepath)

        # For each 6v channel, get max correlation with any 44 channel
        max_corr_per_6v = np.abs(corr_matrix).max(axis=1)  # [256]

        # Mark channels above threshold as "noisy"
        noisy_mask = max_corr_per_6v > self.correlation_threshold
        self.register_buffer(
            "noisy_channel_mask", torch.tensor(noisy_mask, dtype=torch.bool)
        )

    def register_noisy_channels(self, noisy_mask):
        """Manually set which channels are considered noisy."""
        self.register_buffer(
            "noisy_channel_mask", torch.tensor(noisy_mask, dtype=torch.bool)
        )

    def forward(self, x_6v, x_44=None):
        """
        Args:
            x_6v: [B, T, 256] Area 6v features
            x_44: Not used (correlations are precomputed)
        Returns:
            x_aug: [B, T, 256] Features with noisy channels dropped
        """
        if not self.training or self.noisy_channel_mask is None:
            return x_6v

        if self.dropout_prob <= 0:
            return x_6v

        # Create dropout mask: 1 = keep, 0 = drop
        # Higher dropout probability for noisy channels
        mask = torch.ones_like(x_6v)

        # For noisy channels, apply dropout
        noisy_indices = self.noisy_channel_mask.nonzero(as_tuple=True)[0]
        if len(noisy_indices) > 0:
            # Random dropout for noisy channels only
            dropout_mask = (
                torch.rand(
                    x_6v.shape[0],
                    x_6v.shape[1],
                    len(noisy_indices),
                    device=x_6v.device,
                )
                > self.dropout_prob
            )
            mask[:, :, noisy_indices] = dropout_mask.float()

        # Scale by (1 / keep_prob) to maintain expected value
        # Only scale noisy channels that were subject to dropout
        scale = torch.ones_like(x_6v)
        if len(noisy_indices) > 0:
            scale[:, :, noisy_indices] = 1.0 / (1.0 - self.dropout_prob + 1e-8)

        return x_6v * mask * scale


class DualRegionAugmentationPipeline(nn.Module):
    """
    Unified augmentation pipeline for dual-region data using Area 44 for augmentation.

    Combines multiple augmentation strategies:
    - CrossAreaMixup: Mix 6v with aligned 44 signal
    - AdaptiveNoise44: Scale noise by 44's variance
    - CorrelationGuidedDropout: Drop 6v channels correlated with 44
    - Standard white noise and constant offset

    Set parameters to 0 to disable individual strategies.

    Stochastic Modes:
        - stochastic="independent": Each aug applied independently with given probability
        - stochastic="exclusive": Exactly ONE aug selected per sample (no combos, no none)

    Args:
        mixup_ratio: Max ratio for CrossAreaMixup (0 = disabled)
        shuffle_44: If True, shuffle 44 before mixing (control experiment)
        adaptive_noise_base: Base noise for AdaptiveNoise44 (0 = disabled)
        adaptive_noise_scale: Adaptive scale for AdaptiveNoise44
        corr_dropout_prob: Dropout prob for CorrelationGuidedDropout (0 = disabled)
        corr_threshold: Correlation threshold for identifying noisy channels
        corr_file: Path to precomputed correlations
        white_noise_sd: Standard white noise (applied after other augs)
        constant_offset_sd: Constant offset noise
        stochastic: False, "independent", or "exclusive"
        mixup_prob: Probability of applying mixup (independent mode)
        adaptive_prob: Probability of applying adaptive noise (independent mode)
        white_prob: Probability of applying white noise (independent mode)
    """

    def __init__(
        self,
        mixup_ratio=0.0,
        shuffle_44=False,
        adaptive_noise_base=0.0,
        adaptive_noise_scale=0.0,
        corr_dropout_prob=0.0,
        corr_threshold=0.3,
        corr_file=None,
        white_noise_sd=0.8,
        constant_offset_sd=0.2,
        # Stochastic mode parameters
        stochastic=False,  # False, "independent", or "exclusive"
        mixup_prob=0.333,
        adaptive_prob=0.333,
        white_prob=0.333,
    ):
        super().__init__()

        # Stochastic mode: False, "independent", "exclusive", or True (legacy = independent)
        if stochastic is True:
            stochastic = "independent"  # Backward compatibility
        self.stochastic = stochastic
        self.mixup_prob = mixup_prob
        self.adaptive_prob = adaptive_prob
        self.white_prob = white_prob

        # Area 44-based augmentations
        self.mixup = (
            CrossAreaMixup(mix_ratio_max=mixup_ratio, shuffle_44=shuffle_44)
            if mixup_ratio > 0
            else None
        )
        self.adaptive_noise = (
            AdaptiveNoise44(
                base_noise=adaptive_noise_base,
                adaptive_scale=adaptive_noise_scale,
            )
            if adaptive_noise_base > 0 or adaptive_noise_scale > 0
            else None
        )
        self.corr_dropout = (
            CorrelationGuidedDropout(
                dropout_prob=corr_dropout_prob,
                correlation_threshold=corr_threshold,
                correlation_file=corr_file,
            )
            if corr_dropout_prob > 0
            else None
        )

        # Standard augmentations
        self.white_noise_sd = white_noise_sd
        self.constant_offset_sd = constant_offset_sd

    def _apply_exclusive(self, x, x_44):
        """
        Apply exactly ONE augmentation per sample (exclusive mode).
        Each sample randomly selects mixup, adaptive, OR white noise.
        """
        B = x.shape[0]
        device = x.device

        # Build list of available augmentations
        aug_options = []
        if self.mixup is not None and x_44 is not None:
            aug_options.append("mixup")
        if self.adaptive_noise is not None and x_44 is not None:
            aug_options.append("adaptive")
        if self.white_noise_sd > 0:
            aug_options.append("white")

        if len(aug_options) == 0:
            return x

        # Randomly select one aug type per sample
        n_augs = len(aug_options)
        selection = torch.randint(0, n_augs, (B,), device=device)

        # Apply selected augmentation to each sample
        for i, aug_name in enumerate(aug_options):
            mask = selection == i
            if not mask.any():
                continue

            mask_expanded = mask.view(B, 1, 1)

            if aug_name == "mixup":
                x_aug = self.mixup(x, x_44)
                x = torch.where(mask_expanded, x_aug, x)
            elif aug_name == "adaptive":
                x_aug = self.adaptive_noise(x, x_44)
                x = torch.where(mask_expanded, x_aug, x)
            elif aug_name == "white":
                noise = torch.randn_like(x) * self.white_noise_sd
                x = torch.where(mask_expanded, x + noise, x)

        # Constant offset always applied
        if self.constant_offset_sd > 0:
            x = x + (
                torch.randn([B, 1, x.shape[2]], device=device)
                * self.constant_offset_sd
            )

        return x

    def _apply_independent(self, x, x_44):
        """
        Apply augmentations independently per-sample.
        Each sample independently decides whether to apply each augmentation.
        Can result in 0, 1, 2, or 3 augmentations per sample.
        """
        B = x.shape[0]
        device = x.device

        # 1. Mixup (stochastic per-sample)
        if self.mixup is not None and x_44 is not None:
            mixup_mask = torch.rand(B, device=device) < self.mixup_prob
            if mixup_mask.any():
                x_mixed = self.mixup(x, x_44)
                mixup_mask = mixup_mask.view(B, 1, 1)
                x = torch.where(mixup_mask, x_mixed, x)

        # 2. Adaptive noise (stochastic per-sample)
        if self.adaptive_noise is not None and x_44 is not None:
            adaptive_mask = torch.rand(B, device=device) < self.adaptive_prob
            if adaptive_mask.any():
                x_adaptive = self.adaptive_noise(x, x_44)
                adaptive_mask = adaptive_mask.view(B, 1, 1)
                x = torch.where(adaptive_mask, x_adaptive, x)

        # 3. Correlation-guided dropout (always applied if enabled)
        if self.corr_dropout is not None:
            x = self.corr_dropout(x, x_44)

        # 4. White noise (stochastic per-sample)
        if self.white_noise_sd > 0:
            white_mask = torch.rand(B, device=device) < self.white_prob
            if white_mask.any():
                noise = torch.randn_like(x) * self.white_noise_sd
                white_mask = white_mask.view(B, 1, 1)
                x = x + noise * white_mask.float()

        # 5. Constant offset (always applied if enabled)
        if self.constant_offset_sd > 0:
            x = x + (
                torch.randn([B, 1, x.shape[2]], device=device)
                * self.constant_offset_sd
            )

        return x

    def _apply_deterministic(self, x, x_44):
        """
        Apply augmentations deterministically (all enabled augs applied to all samples).
        Original behavior.
        """
        # 1. Mixup first (blends signals)
        if self.mixup is not None and x_44 is not None:
            x = self.mixup(x, x_44)

        # 2. Adaptive noise (adds state-dependent noise)
        if self.adaptive_noise is not None and x_44 is not None:
            x = self.adaptive_noise(x, x_44)

        # 3. Correlation-guided dropout (removes noisy channels)
        if self.corr_dropout is not None:
            x = self.corr_dropout(x, x_44)

        # 4. Standard white noise
        if self.white_noise_sd > 0:
            x = x + torch.randn_like(x) * self.white_noise_sd

        # 5. Constant offset noise
        if self.constant_offset_sd > 0:
            x = x + (
                torch.randn([x.shape[0], 1, x.shape[2]], device=x.device)
                * self.constant_offset_sd
            )

        return x

    def forward(self, x_6v, x_44=None):
        """
        Apply augmentations to Area 6v features, using Area 44 as noise source.

        Args:
            x_6v: [B, T, 256] Area 6v features
            x_44: [B, T, 256] Area 44 features (optional, needed for mixup/adaptive)
        Returns:
            x_aug: [B, T, 256] Augmented 6v features
        """
        if not self.training:
            return x_6v

        if self.stochastic == "exclusive":
            return self._apply_exclusive(x_6v, x_44)
        elif self.stochastic == "independent":
            return self._apply_independent(x_6v, x_44)
        else:
            return self._apply_deterministic(x_6v, x_44)


class TransformerAugmentationPipeline(nn.Module):
    """
    Augmentation pipeline for Transformer models following Feghhi et al. (2025).

    Applies:
    1. White noise (σ=0.2) - lower than GRU (0.8) since time-masking provides regularization
    2. Baseline shift (σ=0.05) - per-electrode DC offset
    3. Causal Gaussian smoothing (kernel=20, σ=2.0) - same as GRU baseline

    Time-masking is handled separately in the Transformer model itself.

    Args:
        white_noise_sd: Standard deviation of additive Gaussian noise (default: 0.2)
        baseline_shift_sd: Standard deviation of per-electrode DC offset (default: 0.05)
        smooth_kernel_size: Kernel size for Gaussian smoothing (default: 20)
        smooth_sigma: Sigma for Gaussian smoothing (default: 2.0)
        neural_dim: Number of neural channels for smoothing (default: 256)
    """

    def __init__(
        self, 
        white_noise_sd=0.2, 
        baseline_shift_sd=0.05,
        smooth_kernel_size=20,
        smooth_sigma=2.0,
        neural_dim=256,
    ):
        super().__init__()
        self.white_noise_sd = white_noise_sd
        self.baseline_shift_sd = baseline_shift_sd
        
        # Causal Gaussian smoothing (same as GRU baseline)
        self.gaussianSmoother = GaussianSmoothing(
            channels=neural_dim, 
            kernel_size=smooth_kernel_size, 
            sigma=smooth_sigma, 
            dim=1
        )

    def forward(self, x, x_44=None):
        """
        Apply augmentations to input features.

        Args:
            x: [B, T, C] neural features
            x_44: Ignored (included for API compatibility with dual-region pipelines)
        Returns:
            x_aug: [B, T, C] augmented features
        """
        if not self.training:
            return x

        # 1. White noise
        if self.white_noise_sd > 0:
            x = x + torch.randn_like(x) * self.white_noise_sd

        # 2. Baseline shift (per-electrode DC offset)
        if self.baseline_shift_sd > 0:
            # Random offset for each electrode, constant across time
            offset = (
                torch.randn(x.shape[0], 1, x.shape[2], device=x.device)
                * self.baseline_shift_sd
            )
            x = x + offset

        # 3. Causal Gaussian smoothing
        # Reshape for conv1d: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.gaussianSmoother(x)
        # Reshape back: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)

        return x
