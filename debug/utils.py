import os

import numpy as np

import librosa
import soundfile as sf



def make_clicked_audio(audio_filepath, times, output_path, audio_weight=1, click_weight=2):
    '''
    How To Use
        from debug.utils import make_clicked_audio
        times = [sixteenth_time_and_counting[0] for i, sixteenth_time_and_counting in enumerate(clipped_sixteenth_times_and_countings) if i%4==0]
        make_clicked_audio(audio_filepath=args.bgm_filepath, times=times, output_path="./testoutput/test1_clicked.wav")
    '''
    audio, sr = librosa.load(audio_filepath)
    click = librosa.clicks(times=times, sr=sr, length=len(audio))
    clicked_audio = mix_audio(audio, click, weight1=audio_weight, weight2=click_weight)
    sf.write(output_path, clicked_audio, sr)




def mix_audio(audio1, audio2, weight1=1.0, weight2=1.0):

    if audio1.ndim == 2:
        audio1 = np.mean(audio1, axis=1)
    if audio2.ndim == 2:
        audio2 = np.mean(audio2, axis=1)

    max_length = max(len(audio1), len(audio2))
    audio1 = np.pad(audio1, (0, max_length - len(audio1)), mode='constant', constant_values=0)
    audio2 = np.pad(audio2, (0, max_length - len(audio2)), mode='constant', constant_values=0)

    mixed_audio = (audio1 * weight1 + audio2 * weight2) / (weight1 + weight2)
    mixed_audio = np.clip(mixed_audio, -1.0, 1.0)

    return mixed_audio
