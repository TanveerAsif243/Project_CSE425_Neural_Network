import numpy as np
import pretty_midi
import os
import random
from collections import defaultdict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

class RandomGenerator:
    def generate(self, length=config.SEQ_LEN):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        current_time = 0
        for _ in range(length):
            pitch = random.randint(config.PIANO_RANGE[0], config.PIANO_RANGE[1]-1)
            duration = random.uniform(0.1, 0.5)
            note = pretty_midi.Note(velocity=80, pitch=pitch, start=current_time, end=current_time + duration)
            piano.notes.append(note)
            current_time += duration
            
        midi.instruments.append(piano)
        return midi

class MarkovChainGenerator:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.pitches = []

    def train(self, midi_files):
        print("Training Markov Chain...")
        for midi_path in midi_files[:10]:
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                for inst in midi_data.instruments:
                    pitches = [n.pitch for n in inst.notes]
                    for i in range(len(pitches) - 1):
                        self.transitions[pitches[i]][pitches[i+1]] += 1
                        self.pitches.append(pitches[i])
            except:
                continue

    def generate(self, length=config.SEQ_LEN):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        if not self.transitions:
            return RandomGenerator().generate(length)
            
        current_pitch = random.choice(self.pitches)
        current_time = 0
        for _ in range(length):
            next_choices = list(self.transitions[current_pitch].keys())
            if not next_choices:
                current_pitch = random.choice(self.pitches)
            else:
                weights = list(self.transitions[current_pitch].values())
                current_pitch = random.choices(next_choices, weights=weights)[0]
                
            duration = random.uniform(0.1, 0.5)
            note = pretty_midi.Note(velocity=80, pitch=current_pitch, start=current_time, end=current_time + duration)
            piano.notes.append(note)
            current_time += duration
            
        midi.instruments.append(piano)
        return midi

if __name__ == "__main__":
    os.makedirs(os.path.join(config.OUTPUTS_DIR, 'generated_midis'), exist_ok=True)
    
    print("Generating Random Baseline...")
    rg = RandomGenerator()
    rg.generate().write(os.path.join(config.OUTPUTS_DIR, 'generated_midis', 'baseline_random.mid'))
    
    print("Generating Markov Baseline...")
    mg = MarkovChainGenerator()
    # Use some raw midi files for training
    import json
    with open(os.path.join(config.RAW_MIDI_DIR, 'maestro-v1.0.0.json'), 'r') as f:
        meta = json.load(f)
    paths = [os.path.join(config.RAW_MIDI_DIR, e['midi_filename']) for e in meta[:20]]
    mg.train(paths)
    mg.generate().write(os.path.join(config.OUTPUTS_DIR, 'generated_midis', 'baseline_markov.mid'))
