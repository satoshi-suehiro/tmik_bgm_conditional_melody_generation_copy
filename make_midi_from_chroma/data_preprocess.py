import os
import bisect
import pickle
import argparse

import madmom
import librosa
import pretty_midi

from tqdm import tqdm
import numpy as np



def make_downbeat_array_by_handinputed_bpm(audio_filepath, start_time, bpm):
    if start_time == 0:
        start_time = 4e-6

    duration = librosa.get_duration(path=audio_filepath)
    time_per_beat = 60.0 / bpm
    beat_times_and_countings = [[start_time + i * time_per_beat, float(1 + i % 4)] for i in range(int(duration // time_per_beat))]
    beat_times_and_countings = np.array(beat_times_and_countings)

    return beat_times_and_countings


def downbeat_estimation(wavfile_path, beats_per_bar_candidates=[3, 4], fps=100):

    # downbeat estimation by madmom
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar_candidates, fps=fps)
    act = madmom.features.downbeats.RNNDownBeatProcessor()(wavfile_path)
    beat_times_and_countings = proc(act)

    return beat_times_and_countings


def make_integrated_chroma(
        audio_filename: str,
        bar_times: list,
        chroma_n_fft: int = 2048,
        chroma_hop_length: int = 512,
    ):

    y, sr = librosa.load(path=audio_filename)
    y =  librosa.util.normalize(y)
    spectrogram = np.abs(librosa.stft(y, n_fft=chroma_n_fft, hop_length=chroma_hop_length))**2
    chroma = librosa.feature.chroma_stft(S=spectrogram, sr=sr)

    # y, sr = librosa.load(path=audio_filename)
    # y =  librosa.util.normalize(y)
    # y_harmonic, _ = librosa.effects.hpss(y)
    # spectrogram = np.abs(librosa.stft(y_harmonic, n_fft=chroma_n_fft, hop_length=chroma_hop_length))**2
    # chroma = librosa.feature.chroma_stft(S=spectrogram, sr=sr)

    integrated_chroma = []

    for i in range(len(bar_times) - 1):
        start = bar_times[i]
        end = bar_times[i+1]

        start_idx = start * sr / chroma_hop_length
        end_idx = end * sr / chroma_hop_length

        start_idx = round(start_idx)
        end_idx = round(end_idx)

        integrated_chroma.append(np.average(chroma[:, start_idx:end_idx], axis=1))

    return integrated_chroma

