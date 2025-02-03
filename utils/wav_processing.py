import os

import librosa
# import madmom
import scipy
import numpy  as np

CHROMA_HOP_LENGTH = 512


class Wav():

    def __init__(self, wavfile_path=None, wav=None, sr=None):

        assert wavfile_path != None or (wav is not None and sr != None), \
            "there should be [wavfile_path] or [wav and sr] when initialize Wav class."

        if wav is not None:
            assert wav.ndim == 1, "Wav class is mono-only at now."

        self.wavfile_path = wavfile_path

        if wavfile_path:
            self.wav, self.sr = self.load_wavfile(wavfile_path=wavfile_path)
        else:
            self.wav = wav
            self.sr = sr

        # self.beat_times = None
        # self.beat_times_and_countings = None
        self.chroma_hop_length = CHROMA_HOP_LENGTH
        harmonic, percussive = librosa.effects.hpss(self.wav)
        self.chroma = librosa.feature.chroma_cqt(y=harmonic, sr=self.sr, hop_length=self.chroma_hop_length)



    @staticmethod
    def load_wavfile(wavfile_path, sr=22050, mono=True):
        wav, sr = librosa.core.load(wavfile_path, sr=sr, mono=mono)
        return wav, sr


    def length(self):
        return self.wav.shape[0]


    def bpm(self, beats_num):
        return 60 * beats_num / self.length() * self.sr


    # def beat_estimation(self, fps=100):
    #     wavfile_path = self.wavfile_path

    #     # if wavfile_path == None, write wav temporarily
    #     if self.wavfile_path == None:
    #         self.write(path="./tmp.wav")
    #         wavfile_path = "./tmp.wav"

    #     # beat estimation by madmom
    #     proc = madmom.features.beats.BeatTrackingProcessor(fps=fps)
    #     act = madmom.features.beats.RNNBeatProcessor()(wavfile_path)
    #     self.beat_times = proc(act)

    #     # remove tmp.wav
    #     if self.wavfile_path == None:
    #         os.remove("./tmp.wav")


    # def downbeat_estimation(self, beats_per_bar_candidates=[3, 4], fps=100):
    #     wavfile_path = self.wavfile_path

    #     # if wavfile_path == None, write wav temporarily
    #     if self.wavfile_path == None:
    #         self.write(path="./tmp.wav")
    #         wavfile_path = "./tmp.wav"

    #     # downbeat estimation by madmom
    #     proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar_candidates, fps=fps)
    #     act = madmom.features.downbeats.RNNDownBeatProcessor()(wavfile_path)
    #     self.beat_times_and_countings = proc(act)

    #     # remove tmp.wav
    #     if self.wavfile_path == None:
    #         os.remove("./tmp.wav")


    def clip(self, clip_start_sec, clip_end_sec):
        # clip wav
        self.wav = self.wav[round(clip_start_sec * self.sr) : round(clip_end_sec * self.sr)]

        # clip beat_times if exists
        if self.beat_times is not None:
            new_beat_times = []
            for beat_time in self.beat_times:
                if clip_start_sec <= beat_time <= clip_end_sec:
                    new_beat_times.append(beat_time - clip_start_sec)
            self.beat_times = np.array(new_beat_times)

        # clip beat_times_and_countings if exists
        if self.beat_times_and_countings is not None:
            new_beat_times_and_countings = []
            for beat_time_and_counting in self.beat_times_and_countings:
                beat_time, counting = beat_time_and_counting
                if clip_start_sec <= beat_time <= clip_end_sec:
                    new_beat_times_and_countings.append(np.array([beat_time - clip_start_sec, counting]))
            self.beat_times_and_countings = np.array(new_beat_times_and_countings)


    def time_stretch(self, rate):
        self.wav = librosa.effects.time_stretch(self.wav, rate=rate)


    def write(self, path):
        scipy.io.wavfile.write(path, rate=self.sr, data=self.wav)
