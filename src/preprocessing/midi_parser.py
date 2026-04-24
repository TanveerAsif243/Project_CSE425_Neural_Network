import os
import json
import numpy as np
import pretty_midi
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def midi_to_piano_roll(midi_path, fs=config.FS):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi_data.get_piano_roll(fs=fs)
        piano_roll = piano_roll[config.PIANO_RANGE[0]:config.PIANO_RANGE[1], :]
        piano_roll = piano_roll.T
        piano_roll = (piano_roll > 0).astype(np.float32)
        return piano_roll
    except Exception as e:
        print(f"Error parsing {midi_path}: {e}")
        return None

def segment_piano_roll(piano_roll, seq_len=config.SEQ_LEN):
    segments = []
    for i in range(0, len(piano_roll) - seq_len, seq_len // 2):
        segments.append(piano_roll[i:i+seq_len])
    return segments

def prepare_dataset():
    metadata_path = os.path.join(config.RAW_MIDI_DIR, 'maestro-v1.0.0.json')
    if not os.path.exists(metadata_path):
        print(f"Metadata not found at {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    all_segments = []
    labels = []
    
    filtered_metadata = [entry for entry in metadata if entry['canonical_composer'] in config.TARGET_COMPOSERS]
    
    max_files_per_composer = 5
    composer_counts = {c: 0 for c in config.TARGET_COMPOSERS}
    
    print("Preprocessing MIDI files...")
    for entry in tqdm(filtered_metadata):
        composer = entry['canonical_composer']
        if composer_counts[composer] >= max_files_per_composer:
            continue
            
        midi_path = os.path.join(config.RAW_MIDI_DIR, entry['midi_filename'])
        pr = midi_to_piano_roll(midi_path)
        
        if pr is not None:
            segs = segment_piano_roll(pr)
            if len(segs) > 0:
                all_segments.extend(segs)
                labels.extend([config.TARGET_COMPOSERS.index(composer)] * len(segs))
                composer_counts[composer] += 1
            
    if not all_segments:
        print("No segments generated.")
        return

    X = np.array(all_segments)
    y = np.array(labels)
    
    print(f"Total segments: {len(X)}")
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(config.PROCESSED_DIR, 'X.npy'), X)
    np.save(os.path.join(config.PROCESSED_DIR, 'y.npy'), y)
    print("Dataset saved to", config.PROCESSED_DIR)

if __name__ == "__main__":
    prepare_dataset()
