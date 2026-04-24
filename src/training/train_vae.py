import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.vae import VAE

def vae_loss_fn(x_hat, x, mu, logvar, beta=1.0):
    recon_loss = nn.MSELoss()(x_hat, x)
    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize KL loss by batch size
    kl_loss /= x.size(0) * x.size(1) * x.size(2)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae():
    # Load data
    X = np.load(os.path.join(config.PROCESSED_DIR, 'X.npy'))
    y = np.load(os.path.join(config.PROCESSED_DIR, 'y.npy'))
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    losses = []
    
    print("Starting Task 2: VAE Multi-Genre Training...")
    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        for batch in dataloader:
            x, genres = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, genres)
            loss, _, _ = vae_loss_fn(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {avg_loss:.4f}")
        
    # Save Loss Curve
    plt.figure()
    plt.plot(losses)
    plt.title('Task 2: VAE Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'plots', 'task2_loss.png'))
    print("Loss curve saved.")
    
    # Save Model
    torch.save(model.state_dict(), os.path.join(config.OUTPUTS_DIR, 'task2_vae.pth'))
    return model, device

def generate_multi_genre_samples(model, device):
    print("Generating Multi-Genre Samples for Task 2...")
    model.eval()
    
    with torch.no_grad():
        for g_idx, composer in enumerate(config.TARGET_COMPOSERS):
            for i in range(2): # 2 samples per "genre"
                z = torch.randn(1, config.LATENT_DIM).to(device)
                genre_tensor = torch.tensor([g_idx]).to(device)
                x_hat = model.decode(z, genre_tensor)
                
                piano_roll = x_hat.squeeze(0).cpu().numpy()
                piano_roll = (piano_roll > 0.5).astype(np.float32)
                
                # Export to MIDI
                import pretty_midi
                midi = pretty_midi.PrettyMIDI()
                piano_instrument = pretty_midi.Instrument(program=0)
                
                full_pr = np.zeros((128, config.SEQ_LEN))
                full_pr[config.PIANO_RANGE[0]:config.PIANO_RANGE[1], :] = piano_roll.T
                
                prev_notes = np.zeros(128)
                for t in range(config.SEQ_LEN):
                    for n in range(128):
                        if full_pr[n, t] > 0 and prev_notes[n] == 0:
                            prev_notes[n] = t
                        elif full_pr[n, t] == 0 and prev_notes[n] > 0:
                            start_time = prev_notes[n] / config.FS
                            end_time = t / config.FS
                            note = pretty_midi.Note(velocity=100, pitch=n, start=start_time, end=end_time)
                            piano_instrument.notes.append(note)
                            prev_notes[n] = 0
                
                midi.instruments.append(piano_instrument)
                composer_name = composer.replace(' ', '_')
                midi.write(os.path.join(config.OUTPUTS_DIR, 'generated_midis', f'task2_{composer_name}_sample_{i+1}.mid'))
    print("Multi-genre samples generated.")

if __name__ == "__main__":
    model, device = train_vae()
    generate_multi_genre_samples(model, device)
