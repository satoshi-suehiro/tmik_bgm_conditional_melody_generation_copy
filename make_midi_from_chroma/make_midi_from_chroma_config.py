beats_per_bar_candidates = [4]
spectrogram_n_fft = 2048
spectrogram_hop_length = 256
chroma_n_fft = 2048
chroma_hop_length = 512
slice_length = 256

max_num_chords = 1000
add_n_argmax_pitch_class = 4

pc2nn = {
    'C': 48,
    'C#': 49,
    'D': 50,
    'Eb': 51,
    'E': 52,
    'F': 53,
    'F#': 54,
    'G': 55,
    'Ab': 56,
    'A': 57,
    'Bb': 58,
    'B': 59
}

data_fixed_velocity = 100
quantization_unit = 16
minimum_length = 16

key_change_prob=0.001
chord_change_prob=0.5
chord_pitch_out_of_key_prob=0.01
chord_note_concentration=100.0

