import torch
from torch import nn

from neural_decoder.augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
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
        use_channels="all",  # "all", "6v_only", or "44_only"
    ):

        super(GRUDecoder, self).__init__()

        # Channel selection - input is always 256, but we may use a subset
        self.use_channels = use_channels
        if use_channels == "all":
            self.input_dim = neural_dim  # 256
            self.channel_slice = slice(None)  # all channels
        elif use_channels == "6v_only":
            self.input_dim = neural_dim // 2  # 128
            self.channel_slice = slice(0, neural_dim // 2)  # first 128
        elif use_channels == "44_only":
            self.input_dim = neural_dim // 2  # 128
            self.channel_slice = slice(neural_dim // 2, None)  # last 128
        else:
            raise ValueError(
                f"use_channels must be 'all', '6v_only', or '44_only', got {use_channels}"
            )

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim  # Original input dim (256)
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth

        self.inputLayerNonlinearity = torch.nn.Softsign()

        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        # Use input_dim for layers that process the (potentially sliced) input
        self.gaussianSmoother = GaussianSmoothing(
            self.input_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        self.dayWeights = torch.nn.Parameter(
            torch.randn(nDays, self.input_dim, self.input_dim)
        )

        self.dayBias = torch.nn.Parameter(
            torch.zeros(nDays, 1, self.input_dim)
        )

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(self.input_dim)

        # GRU layers - use input_dim for the input size
        self.gru_decoder = nn.GRU(
            self.input_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=False,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers - use input_dim
        for x in range(nDays):
            setattr(
                self,
                "inpLayer" + str(x),
                nn.Linear(self.input_dim, self.input_dim),
            )

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(self.input_dim)
            )

        # rnn outputs
        self.fc_decoder_out = nn.Linear(
            hidden_dim, n_classes + 1
        )  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        # Apply channel selection (slice input if using subset of channels)
        neuralInput = neuralInput[:, :, self.channel_slice]

        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        h0 = torch.zeros(
            self.layer_dim,
            transformedNeural.size(0),
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
