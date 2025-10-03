
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential





class PositionalEncoding(nn.Module):

  def __init__(self, d_model , max_len):
    self.d_model = d_model
    self.max_len = max_len
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
     x = x + self.pe[:, :x.size(1),:]
     return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, ff_hidden, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.ffn(x)
        x = self.norm2(x + self.dropout(x2))
        return x


class StockTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=32, seq_len=18, num_heads=2, ff_hidden=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder1 = TransformerEncoderBlock(d_model, ff_hidden, num_heads, dropout)
        self.encoder2 = TransformerEncoderBlock(d_model, ff_hidden, num_heads, dropout)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = x[-1, :, :]
        out = self.regressor(x)
        return out.squeeze(-1)

