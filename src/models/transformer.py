import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size=391, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: (SeqLen, Batch)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        
        output = self.transformer_decoder(tgt, memory=torch.zeros_like(tgt), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # Memory is zero because we are only using decoder for autoregressive generation without encoder
        output = self.fc_out(output)
        return output
