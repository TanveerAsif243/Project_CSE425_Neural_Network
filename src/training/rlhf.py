import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.transformer import MusicTransformer
from preprocessing.tokenizer import MIDITokenizer

def reward_function(tokens):
    # tokens is a list of integers
    if not tokens: return 0.0
    
    reward = 0.0
    
    # 1. Pitch Variety Reward
    pitches = [t for t in tokens if 0 <= t <= 127]
    if len(pitches) > 0:
        variety = len(set(pitches)) / len(pitches)
        reward += variety * 2.0
    
    # 2. Rhythm Reward (avoid too many time shifts)
    shifts = [t for t in tokens if 256 <= t <= 355]
    if len(tokens) > 0:
        shift_ratio = len(shifts) / len(tokens)
        if 0.1 < shift_ratio < 0.4:
            reward += 1.0
        else:
            reward -= 1.0
            
    # 3. Scale Consistency (Simple C-Major/A-Minor check)
    c_major = [0, 2, 4, 5, 7, 9, 11]
    in_scale = sum(1 for p in pitches if p % 12 in c_major)
    if pitches:
        scale_ratio = in_scale / len(pitches)
        reward += scale_ratio * 1.5
        
    return reward

def rlhf_tune():
    device = torch.device('cpu')
    print("Loading pretrained Transformer for RLHF...", flush=True)
    model = MusicTransformer().to(device)
    # Assume we have a pretrained state
    path = os.path.join(config.OUTPUTS_DIR, 'task3_transformer.pth')
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Small LR for tuning
    
    print("Starting Task 4: RLHF Tuning (Policy Gradient)...", flush=True)
    
    for i in range(10): # RL iterations
        # Generate a sample
        model.eval()
        tokens = [389] # BOS
        log_probs = []
        
        for _ in range(50): # Generate 50 tokens
            input_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
            tgt_mask = model.generate_square_subsequent_mask(len(tokens)).to(device)
            logits = model(input_tensor, tgt_mask=tgt_mask)
            
            probs = torch.softmax(logits[-1, 0, :], dim=-1)
            dist = torch.distributions.Categorical(probs)
            token = dist.sample()
            
            tokens.append(token.item())
            log_probs.append(dist.log_prob(token))
            
            if token.item() == 390: break # EOS
            
        reward = reward_function(tokens[1:])
        
        # Policy Gradient Update: J = E[R * log_prob]
        # We want to maximize reward, so minimize -Reward * log_prob
        optimizer.zero_grad()
        policy_loss = []
        for lp in log_probs:
            policy_loss.append(-lp * reward)
        
        total_loss = torch.stack(policy_loss).sum()
        total_loss.backward()
        optimizer.step()
        
        print(f"Iteration {i+1}, Reward: {reward:.4f}", flush=True)
        
    torch.save(model.state_dict(), os.path.join(config.OUTPUTS_DIR, 'task4_rlhf.pth'))
    print("RLHF tuning complete.", flush=True)

if __name__ == "__main__":
    rlhf_tune()
