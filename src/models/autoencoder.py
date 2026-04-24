import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class Encoder(nn.Module):
    def __init__(self, input_dim=config.NUM_PIANO_NOTES, hidden_dim=config.HIDDEN_DIM, latent_dim=config.LATENT_DIM):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (Batch, SeqLen, InputDim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (1, Batch, HiddenDim)
        z = self.fc(h_n.squeeze(0))
        return z

class Decoder(nn.Module):
    def __init__(self, output_dim=config.NUM_PIANO_NOTES, hidden_dim=config.HIDDEN_DIM, latent_dim=config.LATENT_DIM, seq_len=config.SEQ_LEN):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # z: (Batch, LatentDim)
        h = self.fc(z)
        # Repeat h for each time step
        h_repeated = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        # h_repeated: (Batch, SeqLen, HiddenDim)
        out, _ = self.lstm(h_repeated)
        # out: (Batch, SeqLen, HiddenDim)
        x_hat = self.sigmoid(self.out(out))
        return x_hat

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=config.NUM_PIANO_NOTES, hidden_dim=config.HIDDEN_DIM, latent_dim=config.LATENT_DIM, seq_len=config.SEQ_LEN):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
