"""
Phase 2, Step 1: CNN-Transformer for the t -> t+1 diameter forecasting
task, using the same cached ResNet18 embeddings as the CNN-LSTM (Step 6)
so this is a clean architecture swap, not a new data pipeline.

CONTINUOUS POSITIONAL ENCODING (the key Phase 2 design point): standard
Transformer sinusoidal positional encoding assumes evenly-spaced integer
positions (0, 1, 2, ...), which is a poor fit here -- acquisition gaps
are irregular (2-42 days, see README's "Data characteristics" section).
Instead, this implementation computes the sinusoidal encoding using each
timestep's REAL day_after_planting value in place of the sequence index:

    PE(day, 2i)   = sin(day / 10000^(2i/d_model))
    PE(day, 2i+1) = cos(day / 10000^(2i/d_model))

This is parameter-free (no extra weights to learn or overfit), and
generalizes naturally to forecast gaps/day values not seen during
training, unlike a learned positional embedding indexed by integer
position. day_after_planting is used in RAW (unnormalized) day units
here, not the z-scored version used elsewhere as an input feature --
the sinusoidal formula's frequency spectrum is calibrated against
realistic day-count magnitudes (up to ~100 days here), not standardized
values centered near 0.

CAUSALITY: a causal attention mask ensures position i only attends to
positions 0..i (never future timesteps), mirroring the CNN-LSTM's
unidirectional property, since the task is strictly next-step forecasting.
Padding is also masked out so padded positions never contribute to
attention for real tokens.
"""
import math

import torch
import torch.nn as nn


def continuous_sinusoidal_pe(days, d_model):
    """
    days: (B, T) float tensor of RAW day_after_planting values (not
          normalized) -- padded positions may hold arbitrary values,
          since they get masked out separately in the model.
    Returns: (B, T, d_model) positional encoding.
    """
    device = days.device
    batch, seq_len = days.shape
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )  # (d_model/2,)

    days = days.unsqueeze(-1)  # (B, T, 1)
    angles = days * div_term  # (B, T, d_model/2)

    pe = torch.zeros(batch, seq_len, d_model, device=device)
    pe[..., 0::2] = torch.sin(angles)
    pe[..., 1::2] = torch.cos(angles)
    return pe


class CNNTransformer(nn.Module):
    def __init__(self, input_dim=512, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        # Project raw 512-dim ResNet18 embeddings into the transformer's
        # working dimension. day_after_planting is NOT concatenated into
        # this projection (unlike the LSTM, which appends it as a 513th
        # input feature) -- here it drives the positional encoding
        # instead, which is the more natural place for timing information
        # in a Transformer, and avoids conflating "what the plant looks
        # like" with "when this was observed" in a single feature vector.
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, embeddings, days_raw, lengths):
        """
        embeddings: (B, T_max, 512) padded CNN embeddings
        days_raw:   (B, T_max) padded RAW day_after_planting values
        lengths:    (B,) true sequence lengths (long tensor, CPU or GPU)
        """
        batch, seq_len, _ = embeddings.shape
        device = embeddings.device

        x = self.input_proj(embeddings)  # (B, T, d_model)
        pe = continuous_sinusoidal_pe(days_raw, self.d_model)
        x = x + pe

        # Combine the causal mask and the padding mask into a single
        # additive float mask, since PyTorch deprecates mixing a float
        # `mask` with a bool `src_key_padding_mask` in the same call.
        # Padding mask: True at PADDED positions (to be ignored).
        arange = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
        is_pad = arange >= lengths.to(device).unsqueeze(1)  # (B, T), True = pad
        padding_mask_additive = torch.zeros(batch, seq_len, device=device)
        padding_mask_additive.masked_fill_(is_pad, float("-inf"))
        # Broadcast to (B, 1, 1, T) -> expand over the (T_query, T_key) causal
        # mask's key dimension so each query position also can't attend to
        # padded key positions, on top of the causal (future) restriction.
        padding_mask_additive = padding_mask_additive.unsqueeze(1)  # (B, 1, T)

        # Causal mask: position i may attend to positions 0..i only.
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )  # (T, T)

        # Combine: (B, 1, T) + (T, T) broadcasts to (B, T, T), one mask per
        # batch element, expanded per attention head by nn.TransformerEncoder
        # internally when given a 3D mask replicated across heads.
        combined_mask = causal_mask.unsqueeze(0) + padding_mask_additive  # (B, T, T)
        # nn.TransformerEncoderLayer expects either (T, T) or
        # (B * nhead, T, T) for a float mask; repeat per head here.
        nhead = self.encoder.layers[0].self_attn.num_heads
        combined_mask = combined_mask.repeat_interleave(nhead, dim=0)  # (B*nhead, T, T)
        # Rows that are entirely -inf (a fully-padded query position) would
        # produce NaN softmax; zero those rows out since they're never read
        # back (see last_hidden gather below, which only reads true-length
        # positions).
        all_masked = torch.isinf(combined_mask).all(dim=-1)
        combined_mask = combined_mask.masked_fill(all_masked.unsqueeze(-1), 0.0)

        encoded = self.encoder(x, mask=combined_mask)  # (B, T, d_model)

        # Take each sequence's own last TRUE (non-padded) timestep's representation.
        last_idx = (lengths - 1).clamp(min=0).to(device)  # (B,)
        batch_idx = torch.arange(batch, device=device)
        last_hidden = encoded[batch_idx, last_idx]  # (B, d_model)

        out = self.head(last_hidden).squeeze(-1)  # (B,)
        return out
