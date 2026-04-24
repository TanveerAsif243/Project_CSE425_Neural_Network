import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.transformer import MusicTransformer

class TokenDataset(Dataset):
    def __init__(self, token_lists, seq_len=256):
        self.samples = []
        for tl in token_lists:
            for i in range(0, len(tl) - seq_len, seq_len // 2):
                self.samples.append(tl[i:i+seq_len])
        self.samples = torch.LongTensor(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # We need input and target (shifted by 1)
        x = self.samples[idx, :-1]
        y = self.samples[idx, 1:]
        return x, y

def train_transformer():
    print("Loading tokens...", flush=True)
    with open(os.path.join(config.PROCESSED_DIR, 'tokens.pkl'), 'rb') as f:
        token_data = pickle.load(f)
        
    print(f"Loaded {len(token_data)} sequences.", flush=True)
    dataset = TokenDataset(token_data)
    if len(dataset) == 0:
        print("Dataset is empty. Skipping training.", flush=True)
        return
        
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    print(f"Training on {len(dataset)} segmented samples.", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    model = MusicTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    print("Starting Task 3: Transformer Training...", flush=True)
    
    try:
        model.train()
        for epoch in range(5): # Limit for demo
            epoch_loss = 0
            for x, y in dataloader:
                # Transformer expects (SeqLen, Batch)
                x = x.transpose(0, 1).to(device)
                y = y.transpose(0, 1).to(device)
                
                tgt_mask = model.generate_square_subsequent_mask(x.size(0)).to(device)
                
                optimizer.zero_grad()
                output = model(x, tgt_mask=tgt_mask)
                
                # Reshape for CrossEntropy
                loss = criterion(output.view(-1, 391), y.reshape(-1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            perplexity = np.exp(avg_loss)
            print(f"Epoch [{epoch+1}/5], Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}", flush=True)
    except Exception as e:
        print(f"Error during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
    # Save Loss and Perplexity Plots
    plt.figure()
    plt.plot(losses)
    plt.title('Task 3: Transformer CrossEntropy Loss')
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'plots', 'task3_loss.png'))
    
    torch.save(model.state_dict(), os.path.join(config.OUTPUTS_DIR, 'task3_transformer.pth'))
    print("Transformer trained and saved.", flush=True)
    return model, device

if __name__ == "__main__":
    train_transformer()
