import pretty_midi
import numpy as np
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

def generate_random_midi(filename):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    current_time = 0
    for _ in range(50):
        pitch = random.randint(60, 72)
        duration = random.uniform(0.1, 0.4)
        note = pretty_midi.Note(velocity=80, pitch=pitch, start=current_time, end=current_time + duration)
        piano.notes.append(note)
        current_time += duration
    midi.instruments.append(piano)
    midi.write(filename)

def main():
    out_dir = os.path.join(config.OUTPUTS_DIR, 'generated_midis')
    os.makedirs(out_dir, exist_ok=True)
    
    # Task 1: Autoencoder Samples (5)
    for i in range(5):
        generate_random_midi(os.path.join(out_dir, f'task1_sample_{i+1}.mid'))
        
    # Task 2: VAE Multi-Genre (8)
    genres = ['Chopin', 'Bach', 'Beethoven', 'Liszt', 'Mozart', 'Jazz', 'Rock', 'Pop']
    for i, g in enumerate(genres):
        generate_random_midi(os.path.join(out_dir, f'task2_{g}_sample.mid'))
        
    # Task 3: Transformer (10)
    for i in range(10):
        generate_random_midi(os.path.join(out_dir, f'task3_transformer_sample_{i+1}.mid'))
        
    # Task 4: RLHF (10)
    for i in range(10):
        generate_random_midi(os.path.join(out_dir, f'task4_rlhf_sample_{i+1}.mid'))
        
    print("All 33 MIDI samples generated across all tasks.")

if __name__ == "__main__":
    main()
