import os
import sys
import math
import bisect

import pretty_midi

from tqdm import tqdm
import numpy as np

from make_midi_from_chroma.data_preprocess import downbeat_estimation, make_integrated_chroma

from make_midi_from_chroma.magenta_chord_recognition import _key_chord_distribution, _key_chord_transition_distribution, _chord_frame_log_likelihood, _key_chord_viterbi, \
                                    NO_CHORD, _PITCH_CLASS_NAMES, _CHORD_KIND_PITCHES

import make_midi_from_chroma.make_midi_from_chroma_config as config

from make_midi_from_chroma.errors import NotEnoughLengthError


def clip_beat_times_and_countings(beat_times_and_countings):
    while True:
        if beat_times_and_countings[0][1] != 1.0:
            beat_times_and_countings = beat_times_and_countings[1:]
        else:
            break

    while True:
        if beat_times_and_countings[-1][1] != 1.0:
            beat_times_and_countings = beat_times_and_countings[:-1]
        else:
            break

    assert beat_times_and_countings[0][1] == 1.0 and beat_times_and_countings[-1][1] == 1.0, print(beat_times_and_countings)

    return beat_times_and_countings


def make_bar_times(beat_times_and_countings):
    beats_per_bar = config.beats_per_bar_candidates[0]

    assert len(beat_times_and_countings) % beats_per_bar == 1, print(beat_times_and_countings, len(beat_times_and_countings))

    beat_times_and_countings = beat_times_and_countings.tolist()

    bar_times = []
    for i in range(len(beat_times_and_countings) // beats_per_bar + 1):
        bar_times.append(beat_times_and_countings[i*beats_per_bar][0])

    return bar_times


def make_sixteenth_times_and_countings(
        beat_times_and_countings: np.ndarray,
        quantization_unit: int = 16,
        beats_per_bar: int = 4
    ):

    assert quantization_unit in [16, 8, 4]
    assert isinstance(beats_per_bar, int)

    beat_times_and_countings = beat_times_and_countings.tolist()

    # convert countings to programmably easy measure
    for i in range(len(beat_times_and_countings)):
        beat_times_and_countings[i][1] = beat_times_and_countings[i][1] - 1

    # convert countings to cumulative countings
    counting_offset = 0
    for i in range(1, len(beat_times_and_countings)):
        counting = beat_times_and_countings[i][1]

        if counting == 0:
            counting_offset += beats_per_bar

        beat_times_and_countings[i][1] = counting_offset + counting

    # add semi note to beat_times_and_countings
    for i in range(0, len(beat_times_and_countings) - 1):
        beat_time, counting = beat_times_and_countings[i]
        next_beat_time, _ = beat_times_and_countings[i+1]

        if quantization_unit == 16:
            beat_times_and_countings.append([
                beat_time + (next_beat_time - beat_time) / 4.0 * 1,
                counting + 1.0 / 4.0 * 1
            ])
        if quantization_unit in [16, 8]:
            beat_times_and_countings.append([
                beat_time + (next_beat_time - beat_time) / 4.0 * 2,
                counting + 1.0 / 4.0 * 2
            ])
        if quantization_unit == 16:
            beat_times_and_countings.append([
                beat_time + (next_beat_time - beat_time) / 4.0 * 3,
                counting + 1.0 / 4.0 * 3
            ])
    beat_times_and_countings = sorted(beat_times_and_countings, key=lambda x: x[1])

    return beat_times_and_countings


def sequence_note_pitch_vectors(integrated_chroma):

    x = np.zeros([len(integrated_chroma), 12])

    for i, chroma in enumerate(integrated_chroma):

        sorted_chroma = sorted([(pitch_class, value) for (pitch_class, value) in enumerate(chroma)], key = lambda x: x[1], reverse=True)

        for item in sorted_chroma[:config.add_n_argmax_pitch_class]:
            pitch_class, value = item
            x[i, pitch_class] += value

    x_norm = np.linalg.norm(x, axis=1)
    nonzero_frames = x_norm > 0
    x[nonzero_frames, :] /= x_norm[nonzero_frames, np.newaxis]

    return x


def infer_chords_for_sequence(
    integrated_chroma,

    key_chord_loglik=None,
    key_chord_transition_loglik=None,
    key_change_prob=0.001,
    chord_change_prob=0.5,
    chord_pitch_out_of_key_prob=0.01,
    chord_note_concentration=100.0,
):


    if len(integrated_chroma) > config.max_num_chords:
        raise Exception(
            'Too long for chord inference: %d frames' % len(integrated_chroma))

    # Compute pitch vectors for each chord frame, then compute log-likelihood of
    # observing those pitch vectors under each possible chord.
    note_pitch_vectors = sequence_note_pitch_vectors(integrated_chroma=integrated_chroma)

    chord_frame_loglik = _chord_frame_log_likelihood(
        note_pitch_vectors, chord_note_concentration)

    # Compute distribution over chords for each key, and transition distribution
    # between key-chord pairs.
    if key_chord_loglik is None:
        key_chord_distribution = _key_chord_distribution(
            chord_pitch_out_of_key_prob=chord_pitch_out_of_key_prob)
        key_chord_loglik = np.log(key_chord_distribution)

    if key_chord_transition_loglik is None:
        key_chord_transition_distribution = _key_chord_transition_distribution(
            key_chord_distribution,
            key_change_prob=key_change_prob,
            chord_change_prob=chord_change_prob)
        key_chord_transition_loglik = np.log(key_chord_transition_distribution)

    key_chords = _key_chord_viterbi(
        chord_frame_loglik, key_chord_loglik, key_chord_transition_loglik)


    chords = []
    for frame, (key, chord) in enumerate(key_chords):

        if chord == NO_CHORD:
            figure = NO_CHORD
        else:
            root, kind = chord
            figure = '%s:%s' % (_PITCH_CLASS_NAMES[(root) % 12], kind)

        chords.append(figure)

    assert len(chords) == len(integrated_chroma)

    return chords


def make_chord_pitches(chords):
    chord_pitches = []
    for c in chords:
        if c == 'N.C.':
            chord_pitches.append([])
        else:
            r, k = c.split(':')
            root_note_num = config.pc2nn[r]
            pitch_offsets = _CHORD_KIND_PITCHES[k]
            chord_pitches.append([root_note_num + pitch_offset for pitch_offset in pitch_offsets])

    return chord_pitches


def calc_avg_bpm(beat_times_and_countings):

    start_time, _ = beat_times_and_countings[0]
    end_time, _ = beat_times_and_countings[-1]
    avg_sec_per_beat = (end_time - start_time) / (len(beat_times_and_countings) - 1)
    avg_bpm = 60 / avg_sec_per_beat

    return avg_bpm


def quantize_in_temporally_changing_bpm_song(
    midi,
    targets,
    sixteenth_times_and_countings,
    quantization_unit,
    beats_per_bar
):

    if isinstance(targets, str):
        assert targets == "all"
        targets = [i for i in range(len(midi.instruments))]
    elif isinstance(targets, list):
        for target in targets:
            assert isinstance(target, int)
            assert target < len(midi.instruments)
    else:
        assert False, "targets should be 'all' or <list[int]>"

    assert quantization_unit in [16, 8, 4]
    assert isinstance(beats_per_bar, int)

    beat_times = [v[0] for v in sixteenth_times_and_countings]
    countings = [v[1] for v in sixteenth_times_and_countings]

    # quantize
    def find_closest_element_index(sorted_lst, target):
        pos = bisect.bisect_left(sorted_lst, target)

        if pos == 0:
            return 0
        if pos == len(sorted_lst):
            return len(sorted_lst) - 1

        before = pos - 1
        after = pos
        if abs(sorted_lst[after] - target) < abs(sorted_lst[before] - target):
            return after
        else:
            return before

    for target in targets:
        for i, note in enumerate(midi.instruments[target].notes):
            start_closest_index = find_closest_element_index(beat_times, note.start)
            start_closest_counting = countings[start_closest_index]
            start_ticks = int(midi.resolution * start_closest_counting)

            end_closest_index = find_closest_element_index(beat_times, note.end)
            end_closest_counting = countings[end_closest_index]
            end_ticks = int(midi.resolution * end_closest_counting)

            midi.instruments[target].notes[i].start = midi.tick_to_time(start_ticks)
            midi.instruments[target].notes[i].end = midi.tick_to_time(end_ticks)

    return midi


def extend_zero_length_notes(
    midi,
    targets = "all",
    minimum_length = 16
):
    if isinstance(targets, str):
        assert targets == "all"
        targets = [i for i in range(len(midi.instruments))]
    elif isinstance(targets, list):
        for target in targets:
            assert isinstance(target, int)
            assert target < len(midi.instruments)
    else:
        assert False, "targets should be 'all' or <list[int]>"

    assert minimum_length in [16, 8, 4]
    minimum_length_ticks = int(midi.resolution / (minimum_length // 4))

    for target in targets:
        for i, note in enumerate(midi.instruments[target].notes):
            start_tick = midi.time_to_tick(note.start)
            end_tick = midi.time_to_tick(note.end)

            if start_tick == end_tick:
                extended_end_tick = int(end_tick + minimum_length_ticks)
                midi.instruments[target].notes[i].end = midi.tick_to_time(extended_end_tick)

    return midi


def make_data(
    sixteenth_times_and_countings,
    chord_pitches,
    bar_times,
    avg_bpm,
):
    # load chords
    assert len(bar_times) == len(chord_pitches) + 1
    chords_track = pretty_midi.Instrument(program=0, is_drum=False, name="chords")
    for frame in range(len(chord_pitches)):
        for pitch in chord_pitches[frame]:
            note = pretty_midi.Note(
                velocity=config.data_fixed_velocity,
                pitch=pitch,
                start=bar_times[frame],
                end=bar_times[frame+1]
            )
            chords_track.notes.append(note)

    integrated_midi = pretty_midi.PrettyMIDI(initial_tempo = avg_bpm)
    integrated_midi.instruments = [chords_track]

    integrated_midi = quantize_in_temporally_changing_bpm_song(
        midi=integrated_midi,
        targets = "all",
        sixteenth_times_and_countings = sixteenth_times_and_countings,
        quantization_unit = config.quantization_unit,
        beats_per_bar = config.beats_per_bar_candidates[0]
    )
    integrated_midi = extend_zero_length_notes(midi=integrated_midi, targets="all", minimum_length=config.minimum_length)
    integrated_midi.remove_invalid_notes()

    return integrated_midi



def make_midi_from_chroma(audio_file_path):

    beat_times_and_countings = downbeat_estimation(wavfile_path=audio_file_path, beats_per_bar_candidates=config.beats_per_bar_candidates)

    clipped_beat_times_and_countings = clip_beat_times_and_countings(beat_times_and_countings=beat_times_and_countings)
    bar_times = make_bar_times(beat_times_and_countings=clipped_beat_times_and_countings)

    if len(bar_times) <= 1:
        raise NotEnoughLengthError

    clipped_sixteenth_times_and_countings = make_sixteenth_times_and_countings(beat_times_and_countings=clipped_beat_times_and_countings)

    integrated_chroma = make_integrated_chroma(
        audio_filename=audio_file_path,
        bar_times=bar_times,
        chroma_n_fft=config.chroma_n_fft,
        chroma_hop_length=config.chroma_hop_length
    )

    chords = infer_chords_for_sequence(
        integrated_chroma=integrated_chroma,
        key_change_prob=config.key_change_prob,
        chord_change_prob=config.chord_change_prob,
        chord_pitch_out_of_key_prob=config.chord_pitch_out_of_key_prob,
        chord_note_concentration=config.chord_note_concentration
    )
    chord_pitches = make_chord_pitches(chords=chords)

    data = make_data(
        sixteenth_times_and_countings=clipped_sixteenth_times_and_countings,
        chord_pitches=chord_pitches,
        bar_times=bar_times,
        avg_bpm=calc_avg_bpm(beat_times_and_countings=beat_times_and_countings)
    )

    return data, clipped_sixteenth_times_and_countings