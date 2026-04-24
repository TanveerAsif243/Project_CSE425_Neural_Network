import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class VAE(nn.Module):
    def __init__(self, input_dim=config.NUM_PIANO_NOTES, hidden_dim=config.HIDDEN_DIM, latent_dim=config.LATENT_DIM, seq_len=config.SEQ_LEN, num_genres=len(config.TARGET_COMPOSERS)):
        super(VAE, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Genre Embedding for conditioning
        self.genre_emb = nn.Embedding(num_genres, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim) # We'll combine z and genre_emb
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, genre_idx):
        # z: (Batch, LatentDim)
        # genre_idx: (Batch,)
        g = self.genre_emb(genre_idx)
        # Combine z and genre (e.g., addition or concat)
        # Here we'll use addition to keep dim same
        combined = z + g
        
        h = self.decoder_fc(combined)
        h_repeated = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm(h_repeated)
        x_hat = self.sigmoid(self.decoder_out(out))
        return x_hat

    def forward(self, x, genre_idx):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, genre_idx)
        return x_hat, mu, logvar
