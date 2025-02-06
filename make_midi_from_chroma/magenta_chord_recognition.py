# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chord inference for NoteSequences."""
import bisect
import itertools
import math
import numbers

# from absl import logging
import numpy as np

# Names of pitch classes to use (mostly ignoring spelling).
_PITCH_CLASS_NAMES = [
    'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# Pitch classes in a key (rooted at zero).
_KEY_PITCHES = [0, 2, 4, 5, 7, 9, 11]

# Pitch classes in each chord kind (rooted at zero).
_CHORD_KIND_PITCHES = {
    '': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}
_CHORD_KINDS = _CHORD_KIND_PITCHES.keys()
NO_CHORD = 'N.C.'
# All usable chords, including no-chord.
_CHORDS = [NO_CHORD] + list(
    itertools.product(range(12), _CHORD_KINDS))

# All key-chord pairs.
_KEY_CHORDS = list(itertools.product(range(12), _CHORDS))

# Maximum length of chord sequence to infer.
_MAX_NUM_CHORDS = 1000

# MIDI programs that typically sound unpitched.
UNPITCHED_PROGRAMS = (
    list(range(96, 104)) + list(range(112, 120)) + list(range(120, 128)))

# Mapping from time signature to number of chords to infer per bar.
_DEFAULT_TIME_SIGNATURE_CHORDS_PER_BAR = {
    (2, 2): 1,
    (2, 4): 1,
    (3, 4): 1,
    (4, 4): 2,
    (6, 8): 2,
}


def _key_chord_distribution(chord_pitch_out_of_key_prob):
    """Probability distribution over chords for each key."""
    num_pitches_in_key = np.zeros([12, len(_CHORDS)], dtype=np.int32)
    num_pitches_out_of_key = np.zeros([12, len(_CHORDS)], dtype=np.int32)

    # For each key and chord, compute the number of chord notes in the key and the
    # number of chord notes outside the key.
    for key in range(12):
        key_pitches = set((key + offset) % 12 for offset in _KEY_PITCHES)
        for i, chord in enumerate(_CHORDS[1:]):
            root, kind = chord
            chord_pitches = set((root + offset) % 12
                                for offset in _CHORD_KIND_PITCHES[kind])
            num_pitches_in_key[key, i + 1] = len(chord_pitches & key_pitches)
            num_pitches_out_of_key[key, i +
                                   1] = len(chord_pitches - key_pitches)

    # Compute the probability of each chord under each key, normalizing to sum to
    # one for each key.
    mat = ((1 - chord_pitch_out_of_key_prob) ** num_pitches_in_key *
           chord_pitch_out_of_key_prob ** num_pitches_out_of_key)
    mat /= mat.sum(axis=1)[:, np.newaxis]
    return mat


def _key_chord_transition_distribution(
        key_chord_distribution, key_change_prob, chord_change_prob):
    """Transition distribution between key-chord pairs."""
    mat = np.zeros([len(_KEY_CHORDS), len(_KEY_CHORDS)])

    for i, key_chord_1 in enumerate(_KEY_CHORDS):
        key_1, chord_1 = key_chord_1
        chord_index_1 = i % len(_CHORDS)

        for j, key_chord_2 in enumerate(_KEY_CHORDS):
            key_2, chord_2 = key_chord_2
            chord_index_2 = j % len(_CHORDS)

            if key_1 != key_2:
                # Key change. Chord probability depends only on key and not previous
                # chord.
                mat[i, j] = (key_change_prob / 11)
                mat[i, j] *= key_chord_distribution[key_2, chord_index_2]

            else:
                # No key change.
                mat[i, j] = 1 - key_change_prob
                if chord_1 != chord_2:
                    # Chord probability depends on key, but we have to redistribute the
                    # probability mass on the previous chord since we know the chord
                    # changed.
                    mat[i, j] *= (
                        chord_change_prob * (
                            key_chord_distribution[key_2, chord_index_2] +
                            key_chord_distribution[key_2, chord_index_1] / (len(_CHORDS) -
                                                                            1)))
                else:
                    # No chord change.
                    mat[i, j] *= 1 - chord_change_prob

    return mat


def _chord_pitch_vectors():
    """Unit vectors over pitch classes for all chords."""
    x = np.zeros([len(_CHORDS), 12])
    for i, chord in enumerate(_CHORDS[1:]):
        root, kind = chord
        for offset in _CHORD_KIND_PITCHES[kind]:
            x[i + 1, (root + offset) % 12] = 1
    x[1:, :] /= np.linalg.norm(x[1:, :], axis=1)[:, np.newaxis]
    return x


def _chord_frame_log_likelihood(note_pitch_vectors, chord_note_concentration):
    """Log-likelihood of observing each frame of note pitches under each chord."""
    return chord_note_concentration * np.dot(note_pitch_vectors,
                                             _chord_pitch_vectors().T)


def _key_chord_viterbi(chord_frame_loglik,
                       key_chord_loglik,
                       key_chord_transition_loglik):
    """Use the Viterbi algorithm to infer a sequence of key-chord pairs."""
    num_frames, num_chords = chord_frame_loglik.shape
    num_key_chords = len(key_chord_transition_loglik)

    loglik_matrix = np.zeros([num_frames, num_key_chords])
    path_matrix = np.zeros([num_frames, num_key_chords], dtype=np.int32)

    # Initialize with a uniform distribution over keys.
    for i, key_chord in enumerate(_KEY_CHORDS):
        key, unused_chord = key_chord
        chord_index = i % len(_CHORDS)
        loglik_matrix[0, i] = (
            -np.log(12) + key_chord_loglik[key, chord_index] +
            chord_frame_loglik[0, chord_index])

    for frame in range(1, num_frames):
        # At each frame, store the log-likelihood of the best sequence ending in
        # each key-chord pair, along with the index of the parent key-chord pair
        # from the previous frame.
        mat = (np.tile(loglik_matrix[frame - 1][:, np.newaxis],
                       [1, num_key_chords]) +
               key_chord_transition_loglik)
        path_matrix[frame, :] = mat.argmax(axis=0)
        loglik_matrix[frame, :] = (
            mat[path_matrix[frame, :], range(num_key_chords)] +
            np.tile(chord_frame_loglik[frame], 12))

    # Reconstruct the most likely sequence of key-chord pairs.
    path = [np.argmax(loglik_matrix[-1])]
    for frame in range(num_frames, 1, -1):
        path.append(path_matrix[frame - 1, path[-1]])

    return [(index // num_chords, _CHORDS[index % num_chords])
            for index in path[::-1]]


class ChordInferenceError(Exception):  # pylint:disable=g-bad-exception-name
    pass


class SequenceAlreadyHasChordsError(ChordInferenceError):
    pass


class UncommonTimeSignatureError(ChordInferenceError):
    pass


class NonIntegerStepsPerChordError(ChordInferenceError):
    pass


class EmptySequenceError(ChordInferenceError):
    pass


class SequenceTooLongError(ChordInferenceError):
    pass

