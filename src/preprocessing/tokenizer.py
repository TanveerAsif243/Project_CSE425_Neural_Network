import pretty_midi
import numpy as np
import os
import json
import torch
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class MIDITokenizer:
    def __init__(self):
        # Tokens: 
        # 0-127: Note On (pitch)
        # 128-255: Note Off (pitch)
        # 256-355: Time Shift (centiseconds, 10ms to 1s)
        # 356-387: Velocity (bins)
        # 388: PAD
        # 389: BOS
        # 390: EOS
        self.vocab_size = 391
        self.pad_token = 388
        self.bos_token = 389
        self.eos_token = 390
        
    def midi_to_tokens(self, midi_path):
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
        except:
            return None
            
        events = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    events.append({'type': 'note_on', 'pitch': note.pitch, 'time': note.start, 'velocity': note.velocity})
                    events.append({'type': 'note_off', 'pitch': note.pitch, 'time': note.end})
        
        # Sort events by time
        events.sort(key=lambda x: x['time'])
        
        tokens = [self.bos_token]
        last_time = 0
        for event in events:
            # Time shift
            delta_time = event['time'] - last_time
            if delta_time > 0:
                # Max 1 second shift per token
                while delta_time > 1.0:
                    tokens.append(256 + 99) # 1s shift
                    delta_time -= 1.0
                if delta_time > 0.01:
                    shift_token = 256 + int(delta_time * 100)
                    if shift_token < 356:
                        tokens.append(shift_token)
            
            # Event token
            if event['type'] == 'note_on':
                # Velocity (32 bins)
                vel_token = 356 + min(31, event['velocity'] // 4)
                tokens.append(vel_token)
                tokens.append(event['pitch']) # 0-127
            else:
                tokens.append(128 + event['pitch']) # 128-255
                
            last_time = event['time']
            
        tokens.append(self.eos_token)
        return tokens

    def tokens_to_midi(self, tokens):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        current_time = 0
        active_notes = {} # pitch -> start_time
        current_velocity = 100
        
        for token in tokens:
            if 0 <= token <= 127: # Note On
                if token not in active_notes:
                    active_notes[token] = current_time
            elif 128 <= token <= 255: # Note Off
                pitch = token - 128
                if pitch in active_notes:
                    start_time = active_notes.pop(pitch)
                    note = pretty_midi.Note(velocity=current_velocity, pitch=pitch, start=start_time, end=current_time)
                    piano.notes.append(note)
            elif 256 <= token <= 355: # Time Shift
                current_time += (token - 256 + 1) / 100.0
            elif 356 <= token <= 387: # Velocity
                current_velocity = (token - 356) * 4 + 2
                
        midi.instruments.append(piano)
        return midi

def prepare_token_dataset():
    tokenizer = MIDITokenizer()
    metadata_path = os.path.join(config.RAW_MIDI_DIR, 'maestro-v1.0.0.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    token_data = []
    filtered_metadata = [entry for entry in metadata if entry['canonical_composer'] in config.TARGET_COMPOSERS]
    
    print("Tokenizing MIDI files...")
    for entry in tqdm(filtered_metadata[:50]): # Limit for demo
        midi_path = os.path.join(config.RAW_MIDI_DIR, entry['midi_filename'])
        tokens = tokenizer.midi_to_tokens(midi_path)
        if tokens:
            token_data.append(tokens)
            
    # Save as pickle or json
    import pickle
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(config.PROCESSED_DIR, 'tokens.pkl'), 'wb') as f:
        pickle.dump(token_data, f)
    print("Tokens saved.")

if __name__ == "__main__":
    prepare_token_dataset()
