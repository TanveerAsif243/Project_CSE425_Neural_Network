import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_MIDI_DIR = r'e:\CSE_425_Project\maestro-v1.0.0-midi\maestro-v1.0.0'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Preprocessing Constants
FS = 8  # Sampling frequency for piano roll (8 steps per quarter note)
SEQ_LEN = 128  # Fixed sequence length
NUM_NOTES = 128  # MIDI MIDI range
PIANO_RANGE = (21, 109) # A0 to C8 (Standard 88-key piano)
NUM_PIANO_NOTES = PIANO_RANGE[1] - PIANO_RANGE[0]

# Model Constants
LATENT_DIM = 64
HIDDEN_DIM = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10 # For demo/deadline purposes

# Genres/Styles (Using Composers as proxy)
TARGET_COMPOSERS = ['Frédéric Chopin', 'Johann Sebastian Bach', 'Ludwig van Beethoven', 'Franz Liszt', 'Wolfgang Amadeus Mozart']
