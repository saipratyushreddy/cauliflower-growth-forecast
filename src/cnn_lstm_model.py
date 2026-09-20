"""
Step 6: Unidirectional (causal) LSTM over cached CNN embeddings + day-after-
planting, predicting the t+1 trait value.

Causal by construction: torch.nn.LSTM with bidirectional=False (default)
only ever looks at timesteps 0..t when producing its state at t, and the
final hidden state used for prediction is taken at each sequence's own true
length (via pack_padded_sequence), never from padded future positions.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CNNLSTM(nn.Module):
    def __init__(self, input_dim=513, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # causal: never sees future timesteps
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, lengths):
        # features: (B, T_max, input_dim), lengths: (B,)
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]  # (B, hidden_dim) -- final layer's hidden state at each sequence's true last step
        out = self.head(last_hidden).squeeze(-1)  # (B,)
        return out
