import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSequenceClassifier(nn.Module):
    """Hybrid Conv1d + sequence model with optional static features."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        static_size: int = 0,
        conv_channels: int = 32,
        conv_dilations=(1, 2, 4),
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
        sequence_module: str = "lstm",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sequence_module = sequence_module.lower()

        # --- Convolutional stack ---
        in_ch = input_size
        self.convs = nn.ModuleList()
        for d in conv_dilations:
            padding = d
            self.convs.append(
                nn.Conv1d(in_ch, conv_channels, kernel_size=3, padding=padding, dilation=d)
            )
            in_ch = conv_channels

        # --- Sequence model ---
        if self.sequence_module == "lstm":
            self.sequence = nn.LSTM(
                conv_channels,
                rnn_hidden,
                rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
            )
            seq_output = rnn_hidden
        elif self.sequence_module == "gru":
            self.sequence = nn.GRU(
                conv_channels,
                rnn_hidden,
                rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
            )
            seq_output = rnn_hidden
        elif self.sequence_module == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=conv_channels, nhead=4, batch_first=True, dropout=dropout
            )
            self.sequence = nn.TransformerEncoder(encoder_layer, num_layers=rnn_layers)
            seq_output = conv_channels
        else:  # simple pooling
            self.sequence = None
            seq_output = conv_channels

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(seq_output + static_size, rnn_hidden)
        self.fc2 = nn.Linear(rnn_hidden, num_classes)

    def forward(self, seq, static=None):
        """Forward pass.

        Parameters
        ----------
        seq: Tensor
            Shape [batch, time, features]
        static: Tensor or None
            Optional static features of shape [batch, static_size]
        """
        x = seq.transpose(1, 2)  # [batch, features, time]
        for conv in self.convs:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)  # [batch, time, channels]

        if isinstance(self.sequence, nn.LSTM):
            _, (h, _) = self.sequence(x)
            x = h[-1]
        elif isinstance(self.sequence, nn.GRU):
            _, h = self.sequence(x)
            x = h[-1]
        elif isinstance(self.sequence, nn.Module):  # transformer encoder
            x = self.sequence(x)
            x = x.mean(dim=1)
        else:  # pooling
            x = x.mean(dim=1)

        if static is not None:
            x = torch.cat([x, static], dim=1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
