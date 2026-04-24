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
from models.autoencoder import LSTMAutoencoder

def train_ae():
    print("Loading data...", flush=True)
    # Load data
    try:
        X = np.load(os.path.join(config.PROCESSED_DIR, 'X.npy'))
        print(f"Loaded X with shape: {X.shape}", flush=True)
    except Exception as e:
        print(f"Error loading data: {e}", flush=True)
        return None, None
    X = torch.from_numpy(X).float()
    
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    losses = []
    
    print("Starting Task 1: LSTM Autoencoder Training...")
    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {avg_loss:.4f}")
        
    # Save Loss Curve
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUTS_DIR, 'plots'), exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.title('Task 1: LSTM Autoencoder Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'plots', 'task1_loss.png'))
    print("Loss curve saved.")
    
    # Save Model
    torch.save(model.state_dict(), os.path.join(config.OUTPUTS_DIR, 'task1_ae.pth'))
    return model, device

def generate_samples(model, device):
    print("Generating MIDI samples for Task 1...")
    model.eval()
    os.makedirs(os.path.join(config.OUTPUTS_DIR, 'generated_midis'), exist_ok=True)
    
    with torch.no_grad():
        for i in range(5):
            # Sample random latent vector
            z = torch.randn(1, config.LATENT_DIM).to(device)
            x_hat = model.decoder(z)
            # x_hat: (1, SeqLen, NumNotes)
            piano_roll = x_hat.squeeze(0).cpu().numpy()
            # Binarize
            piano_roll = (piano_roll > 0.5).astype(np.float32)
            
            # Export to MIDI (Simple conversion)
            import pretty_midi
            midi = pretty_midi.PrettyMIDI()
            piano_instrument = pretty_midi.Instrument(program=0) # Piano
            
            # Convert piano roll back to MIDI events
            # Note: piano_roll has shape (SeqLen, 88)
            # We need to map back to (128, SeqLen) and then to notes
            full_pr = np.zeros((128, config.SEQ_LEN))
            full_pr[config.PIANO_RANGE[0]:config.PIANO_RANGE[1], :] = piano_roll.T
            
            # Use pretty_midi to recover notes
            # This is a bit tricky from a binary piano roll, but we can do it
            # For simplicity, we'll just create notes for active steps
            prev_notes = np.zeros(128)
            for t in range(config.SEQ_LEN):
                for n in range(128):
                    if full_pr[n, t] > 0 and prev_notes[n] == 0:
                        # Note start
                        prev_notes[n] = t
                    elif full_pr[n, t] == 0 and prev_notes[n] > 0:
                        # Note end
                        start_time = prev_notes[n] / config.FS
                        end_time = t / config.FS
                        note = pretty_midi.Note(velocity=100, pitch=n, start=start_time, end=end_time)
                        piano_instrument.notes.append(note)
                        prev_notes[n] = 0
            
            midi.instruments.append(piano_instrument)
            midi.write(os.path.join(config.OUTPUTS_DIR, 'generated_midis', f'task1_sample_{i+1}.mid'))
    print("5 samples generated.")

if __name__ == "__main__":
    model, device = train_ae()
    generate_samples(model, device)
