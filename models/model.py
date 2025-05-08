import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size, hidden_sizes[0], batch_first=True, bidirectional=False
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_sizes[1], 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, (h_n, _) = self.lstm2(x)
        x = self.dropout2(h_n[-1])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
