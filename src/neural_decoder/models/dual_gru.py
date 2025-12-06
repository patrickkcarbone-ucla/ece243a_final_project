import torch
from torch import nn

from neural_decoder.augmentations import GaussianSmoothing


class DualStreamGRUDecoder(nn.Module):
    """
    Dual-stream GRU decoder that processes Area 6v and Area 44 separately before fusion.
    
    Expects 512-channel input from dual-region dataset:
      - Channels 0-255: Area 6v (128 tx1 + 128 spikePow from ventral premotor)
      - Channels 256-511: Area 44 (128 tx1 + 128 spikePow from Broca's area)
    
    Supports multiple fusion strategies and day-layer configurations for
    ablation testing.
    """
    
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        fusion_strategy="concat",  # "concat", "gated", or "attention"
        day_layer_mode="shared",   # "shared" or "separate"
    ):
        super(DualStreamGRUDecoder, self).__init__()

        # Validate parameters
        if fusion_strategy not in ["concat", "gated", "attention"]:
            raise ValueError(f"fusion_strategy must be 'concat', 'gated', or 'attention', got {fusion_strategy}")
        if day_layer_mode not in ["shared", "separate"]:
            raise ValueError(f"day_layer_mode must be 'shared' or 'separate', got {day_layer_mode}")
        if neural_dim != 512:
            raise ValueError(f"DualStreamGRUDecoder expects neural_dim=512 (256 per region), got {neural_dim}")

        # Store parameters
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.stream_dim = neural_dim // 2  # 256 channels per stream (128 tx1 + 128 spikePow)
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.fusion_strategy = fusion_strategy
        self.day_layer_mode = day_layer_mode

        self.inputLayerNonlinearity = torch.nn.Softsign()

        # Gaussian smoothers for each stream
        self.gaussianSmoother_6v = GaussianSmoothing(
            self.stream_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.gaussianSmoother_44 = GaussianSmoothing(
            self.stream_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        # Unfolders for each stream
        self.unfolder_6v = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.unfolder_44 = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        # Day-specific layers
        self._init_day_layers(nDays)

        # GRU hidden dimension per stream
        # For concat: each GRU has hidden_dim//2, concatenated = hidden_dim
        # For gated: each GRU has hidden_dim, gate combines them to hidden_dim
        # For attention: each GRU has hidden_dim//2, after attention concat = hidden_dim
        if fusion_strategy == "gated":
            self.stream_hidden_dim = hidden_dim
        else:
            self.stream_hidden_dim = hidden_dim // 2

        # GRU for Area 6v stream
        self.gru_6v = nn.GRU(
            self.stream_dim * self.kernelLen,
            self.stream_hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout if layer_dim > 1 else 0,
            bidirectional=False,
        )

        # GRU for Area 44 stream
        self.gru_44 = nn.GRU(
            self.stream_dim * self.kernelLen,
            self.stream_hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout if layer_dim > 1 else 0,
            bidirectional=False,
        )

        # Initialize GRU weights
        for gru in [self.gru_6v, self.gru_44]:
            for name, param in gru.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)

        # Fusion-specific layers
        self._init_fusion_layers()

        # Output layer
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def _init_day_layers(self, nDays):
        """Initialize day-specific transformation layers."""
        if self.day_layer_mode == "shared":
            # Single set of day weights for full 256-dim input
            self.dayWeights = torch.nn.Parameter(
                torch.randn(nDays, self.neural_dim, self.neural_dim)
            )
            self.dayBias = torch.nn.Parameter(
                torch.zeros(nDays, 1, self.neural_dim)
            )
            for x in range(nDays):
                self.dayWeights.data[x, :, :] = torch.eye(self.neural_dim)
        else:
            # Separate day weights for each stream
            self.dayWeights_6v = torch.nn.Parameter(
                torch.randn(nDays, self.stream_dim, self.stream_dim)
            )
            self.dayBias_6v = torch.nn.Parameter(
                torch.zeros(nDays, 1, self.stream_dim)
            )
            self.dayWeights_44 = torch.nn.Parameter(
                torch.randn(nDays, self.stream_dim, self.stream_dim)
            )
            self.dayBias_44 = torch.nn.Parameter(
                torch.zeros(nDays, 1, self.stream_dim)
            )
            for x in range(nDays):
                self.dayWeights_6v.data[x, :, :] = torch.eye(self.stream_dim)
                self.dayWeights_44.data[x, :, :] = torch.eye(self.stream_dim)

    def _init_fusion_layers(self):
        """Initialize fusion-specific layers."""
        if self.fusion_strategy == "gated":
            # Gate layer takes concatenated hidden states, outputs gate values
            self.gate_layer = nn.Sequential(
                nn.Linear(self.stream_hidden_dim * 2, self.stream_hidden_dim),
                nn.Sigmoid()
            )
        elif self.fusion_strategy == "attention":
            # Cross-attention layers
            # Each stream attends to the other
            self.cross_attn_6v = nn.MultiheadAttention(
                embed_dim=self.stream_hidden_dim,
                num_heads=4,
                dropout=self.dropout,
                batch_first=True,
            )
            self.cross_attn_44 = nn.MultiheadAttention(
                embed_dim=self.stream_hidden_dim,
                num_heads=4,
                dropout=self.dropout,
                batch_first=True,
            )
        # For "concat", no additional layers needed

    def _apply_day_transform_shared(self, neuralInput, dayIdx):
        """Apply shared day transformation to full input, then split."""
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        dayBias = torch.index_select(self.dayBias, 0, dayIdx)
        transformed = torch.einsum("btd,bdk->btk", neuralInput, dayWeights) + dayBias
        transformed = self.inputLayerNonlinearity(transformed)
        
        # Split into streams
        stream_6v = transformed[:, :, :self.stream_dim]
        stream_44 = transformed[:, :, self.stream_dim:]
        return stream_6v, stream_44

    def _apply_day_transform_separate(self, stream_6v, stream_44, dayIdx):
        """Apply separate day transformations to each stream."""
        # Area 6v transformation
        dayWeights_6v = torch.index_select(self.dayWeights_6v, 0, dayIdx)
        dayBias_6v = torch.index_select(self.dayBias_6v, 0, dayIdx)
        stream_6v = torch.einsum("btd,bdk->btk", stream_6v, dayWeights_6v) + dayBias_6v
        stream_6v = self.inputLayerNonlinearity(stream_6v)
        
        # Area 44 transformation
        dayWeights_44 = torch.index_select(self.dayWeights_44, 0, dayIdx)
        dayBias_44 = torch.index_select(self.dayBias_44, 0, dayIdx)
        stream_44 = torch.einsum("btd,bdk->btk", stream_44, dayWeights_44) + dayBias_44
        stream_44 = self.inputLayerNonlinearity(stream_44)
        
        return stream_6v, stream_44

    def _apply_gaussian_smoothing(self, stream_6v, stream_44):
        """Apply Gaussian smoothing to each stream."""
        # Permute to [B, C, T] for conv1d
        stream_6v = torch.permute(stream_6v, (0, 2, 1))
        stream_6v = self.gaussianSmoother_6v(stream_6v)
        stream_6v = torch.permute(stream_6v, (0, 2, 1))
        
        stream_44 = torch.permute(stream_44, (0, 2, 1))
        stream_44 = self.gaussianSmoother_44(stream_44)
        stream_44 = torch.permute(stream_44, (0, 2, 1))
        
        return stream_6v, stream_44

    def _apply_unfold(self, stream_6v, stream_44):
        """Apply unfolding (strided windowing) to each stream."""
        # Reshape for Unfold: [B, T, C] -> [B, C, T, 1]
        stream_6v = torch.unsqueeze(torch.permute(stream_6v, (0, 2, 1)), 3)
        stream_44 = torch.unsqueeze(torch.permute(stream_44, (0, 2, 1)), 3)
        
        # Unfold and permute: [B, C*kernelLen, T'] -> [B, T', C*kernelLen]
        stream_6v = torch.permute(self.unfolder_6v(stream_6v), (0, 2, 1))
        stream_44 = torch.permute(self.unfolder_44(stream_44), (0, 2, 1))
        
        return stream_6v, stream_44

    def _apply_gru(self, stream_6v, stream_44, batch_size):
        """Apply GRU to each stream."""
        # Initialize hidden states
        h0_6v = torch.zeros(
            self.layer_dim, batch_size, self.stream_hidden_dim, device=self.device
        ).requires_grad_()
        h0_44 = torch.zeros(
            self.layer_dim, batch_size, self.stream_hidden_dim, device=self.device
        ).requires_grad_()
        
        # Process each stream
        h_6v, _ = self.gru_6v(stream_6v, h0_6v.detach())
        h_44, _ = self.gru_44(stream_44, h0_44.detach())
        
        return h_6v, h_44

    def _apply_fusion(self, h_6v, h_44):
        """Apply fusion strategy to combine stream outputs."""
        if self.fusion_strategy == "concat":
            # Simple concatenation: [B, T', hidden_dim//2] + [B, T', hidden_dim//2] -> [B, T', hidden_dim]
            h_fused = torch.cat([h_6v, h_44], dim=-1)
            
        elif self.fusion_strategy == "gated":
            # Gated fusion: learn to weight contributions from each stream
            concat = torch.cat([h_6v, h_44], dim=-1)
            gate = self.gate_layer(concat)
            h_fused = gate * h_6v + (1 - gate) * h_44
            
        elif self.fusion_strategy == "attention":
            # Cross-attention: each stream attends to the other
            h_6v_enhanced, _ = self.cross_attn_6v(
                query=h_6v, key=h_44, value=h_44
            )
            h_44_enhanced, _ = self.cross_attn_44(
                query=h_44, key=h_6v, value=h_6v
            )
            # Residual connection + concatenation
            h_6v_out = h_6v + h_6v_enhanced
            h_44_out = h_44 + h_44_enhanced
            h_fused = torch.cat([h_6v_out, h_44_out], dim=-1)
        
        return h_fused

    def forward(self, neuralInput, dayIdx):
        """
        Forward pass for dual-stream decoder.
        
        Args:
            neuralInput: [batch, time, 512] neural features from dual-region dataset
                - channels 0-255: Area 6v (tx1 + spikePow)
                - channels 256-511: Area 44 (tx1 + spikePow)
            dayIdx: [batch] day indices for each sample
            
        Returns:
            seq_out: [batch, time', n_classes+1] output logits
        """
        batch_size = neuralInput.size(0)
        
        # Split input into streams at channel 256
        stream_6v = neuralInput[:, :, :self.stream_dim]   # [B, T, 256] - Area 6v
        stream_44 = neuralInput[:, :, self.stream_dim:]   # [B, T, 256] - Area 44
        
        # Apply Gaussian smoothing
        stream_6v, stream_44 = self._apply_gaussian_smoothing(stream_6v, stream_44)
        
        # Apply day-specific transformations
        if self.day_layer_mode == "shared":
            # Recombine for shared transformation, then split again
            combined = torch.cat([stream_6v, stream_44], dim=-1)
            stream_6v, stream_44 = self._apply_day_transform_shared(combined, dayIdx)
        else:
            stream_6v, stream_44 = self._apply_day_transform_separate(
                stream_6v, stream_44, dayIdx
            )
        
        # Apply unfolding (strided windows)
        stream_6v, stream_44 = self._apply_unfold(stream_6v, stream_44)
        
        # Apply GRUs
        h_6v, h_44 = self._apply_gru(stream_6v, stream_44, batch_size)
        
        # Apply fusion
        h_fused = self._apply_fusion(h_6v, h_44)
        
        # Output projection
        seq_out = self.fc_decoder_out(h_fused)
        
        return seq_out

