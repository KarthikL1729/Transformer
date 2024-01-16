# Positional encoding using sine and cosine functions, giving constrained values
# Periodicity also helps in attention

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self):
        pe = torch.zeros(self.max_len, self.d_model)
        even_indices = torch.arange(0, self.max_len, 2).float()
        den = torch.pow(10000, even_indices / d_model)
        pos = torch.arange(0, self.max_len).float().unsqueeze(1)
        even_pe = torch.sin(pos / den)
        odd_pe = torch.cos(pos / den)
        stacked = torch.stack([even_pe, odd_pe], dim=2)
        pe = torch.flatten(stacked, start_dim=1, end_dim=2)
        return pe

max_seq_len = 10
d_model = 6
pe = PositionalEncoding(d_model, max_seq_len)
print(pe.forward()) 