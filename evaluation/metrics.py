import numpy as np
import pretty_midi
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

def get_pitch_histogram(midi_data):
    # (12 pitches)
    histogram = np.zeros(12)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            histogram[note.pitch % 12] += 1
    if histogram.sum() > 0:
        histogram /= histogram.sum()
    return histogram

def pitch_histogram_similarity(h1, h2):
    return 1.0 - np.sum(np.abs(h1 - h2)) / 2.0

def rhythm_diversity_score(midi_data):
    durations = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            durations.append(round(note.end - note.start, 2))
    if not durations:
        return 0.0
    unique_durations = len(set(durations))
    return unique_durations / len(durations)

def repetition_ratio(midi_data, window_size=4):
    # Look for repeated pitch patterns
    pitches = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitches.append(note.pitch)
    
    if len(pitches) < window_size * 2:
        return 0.0
        
    patterns = []
    for i in range(len(pitches) - window_size):
        patterns.append(tuple(pitches[i:i+window_size]))
        
    unique_patterns = len(set(patterns))
    # Ratio of repeated patterns
    return (len(patterns) - unique_patterns) / len(patterns)

def evaluate_midi(midi_path, reference_midi_path=None):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    metrics = {
        'rhythm_diversity': rhythm_diversity_score(midi_data),
        'repetition_ratio': repetition_ratio(midi_data)
    }
    
    if reference_midi_path:
        ref_midi = pretty_midi.PrettyMIDI(reference_midi_path)
        h1 = get_pitch_histogram(midi_data)
        h2 = get_pitch_histogram(ref_midi)
        metrics['pitch_similarity'] = pitch_histogram_similarity(h1, h2)
        
    return metrics

if __name__ == "__main__":
    # Test on a generated sample if exists
    sample_path = os.path.join(config.OUTPUTS_DIR, 'generated_midis', 'task1_sample_1.mid')
    if os.path.exists(sample_path):
        print(f"Evaluating {sample_path}...")
        results = evaluate_midi(sample_path)
        print(results)
