import os
import json


class Informations():

    def __init__(self, jsonfile_path: str):

        self.jsonfile_path = jsonfile_path
        self.informations: list[Information] = []

        if os.path.exists(jsonfile_path):
            with open(jsonfile_path) as f:
                dicts = json.load(f)
                for dict in dicts:
                    self.informations.append(Information(**dict))


    def write(self, indent: int = 2, ensure_ascii: bool = False):
        json_contents = []
        for information in self.informations:
            json_contents.append(vars(information))

        with open(self.jsonfile_path, "w") as f:
            json.dump(json_contents, f, indent=indent, ensure_ascii=ensure_ascii)


class Information():
    def __init__(
            self,
            backmusic_midifile_path: str | None = None,
            backmusic_wavfile_path: str | None = None,
            backmusic_wavfile_sr: int | None = None,
            bars_num: int | None = None,
            beats_per_bar: int | None = None,
            bpm: float | None = None,
            melodies: dict | None = None,
            # melody_midifile_path: str | None = None,
            # melody_wavfile_path: str | None = None,
            # melody_wavfile_sr: int | None = None,
            # mixed_wavfile_path: str | None = None,
            # mixed_wavfile_sr: int | None = None,
            prompts: str | None = None,
            used_melody_seeds: list | None = None,
            **kwargs
    ):
        self.backmusic_midifile_path = backmusic_midifile_path
        self.backmusic_wavfile_path = backmusic_wavfile_path
        self.backmusic_wavfile_sr = backmusic_wavfile_sr
        self.bars_num = bars_num
        self.beats_per_bar = beats_per_bar
        self.bpm = bpm
        self.melodies = melodies
        # self.melody_midifile_path = melody_midifile_path
        # self.melody_wavfile_path = melody_wavfile_path
        # self.melody_wavfile_sr = melody_wavfile_sr
        # self.mixed_wavfile_path = mixed_wavfile_path
        # self.mixed_wavfile_sr = mixed_wavfile_sr
        self.prompts = prompts
        self.used_melody_seeds = used_melody_seeds

