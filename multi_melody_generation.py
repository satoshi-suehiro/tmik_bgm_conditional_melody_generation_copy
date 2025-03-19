import argparse
import os
import json
import copy
import shutil
import random
import datetime
import warnings
import time
import torch
from getmusic.modeling.build import build_model
from getmusic.data.build import build_dataloader
from getmusic.utils.misc import seed_everything, merge_opts_to_config, modify_config_for_debug
from getmusic.utils.io import load_yaml_config
from getmusic.engine.logger import Logger
from getmusic.engine.solver import Solver
from getmusic.distributed.launch import launch
import numpy  as np
import librosa
import soundfile as sf
import pickle
import miditoolkit
import math
from getmusic.utils.midi_config import *
from getmusic.utils.magenta_chord_recognition import infer_chords_for_sequence, _key_chord_distribution,\
    _key_chord_transition_distribution, _CHORDS, _PITCH_CLASS_NAMES, NO_CHORD

import sys
from pretty_midi import Instrument
from utils.information import Information, Informations
from utils.midi_processing import CustomPrettyMIDI
from utils.wav_processing import Wav

from make_midi_from_chroma.make_midi_from_chroma import make_midi_from_chroma, make_midi_by_madmom_chord_recognition


TMP_MIDIFILE = "./tmp.mid"
OUTPUT_SR = 44100
MELODY_WEIGHT = 0.5
MIN_GEN_SEED = 0
MAX_GEN_SEED = 2**32 - 1
INFER_SPLIT_SIZE = 512
AUDIO_FILENAME_EXTENSIONS = [".wav", ".mp3"]
MIDI_FILENAME_EXTENSIONS = [".midi", ".mid"]
JSON_FILENAME_EXTENSIONS = [".json"]

SIXTEENTH_TIMES_AND_COUNTINGS_OUTPUT_FILENAME = "sixteenth_times_and_countings.json"
CONDITONAL_CHORDS_OUTPUT_FILENAME = "conditional_chords.mid"
MELODY_MIDI_OUTPUT_FILENAME = "melody.mid"
MELODY_AUDIO_OUTPUT_FILENAME = "melody.wav"
MIXED_AUDIO_OUTPUT_FILENAME = "mix.wav"

NODE_RANK = os.environ['INDEX'] if 'INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = (os.environ['CHIEF_IP'], 22275) if 'CHIEF_IP' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
NUM_NODE = os.environ['HOST_NUM'] if 'HOST_NUM' in os.environ else 1

inst_to_row = { '80':0, '32':1, '128':2,  '25':3, '0':4, '48':5, '129':6}
prog_to_abrv = {'0':'P','25':'G','32':'B','48':'S','80':'M','128':'D'}
track_name = ['lead', 'bass', 'drum', 'guitar', 'piano', 'string']

c2info = {
    "l": {"track_id": 0, "add_to_conditional_inst": True, "program_num": '80'},
    "b": {"track_id": 1, "add_to_conditional_inst": True, "program_num": '32'},
    "d": {"track_id": 2, "add_to_conditional_inst": False, "program_num": None},
    "g": {"track_id": 3, "add_to_conditional_inst": True, "program_num": '25'},
    "p": {"track_id": 4, "add_to_conditional_inst": True, "program_num": '0'},
    "s": {"track_id": 5, "add_to_conditional_inst": True, "program_num": '48'},
    "c": {"track_id": 6, "add_to_conditional_inst": False, "program_num": None}
}

root_dict = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
kind_dict = {'null': 0, 'm': 1, '+': 2, 'dim': 3, 'seven': 4, 'maj7': 5, 'm7': 6, 'm7b5': 7}
root_list = list(root_dict.keys())
kind_list = list(kind_dict.keys())

_CHORD_KIND_PITCHES = {
    'null': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    'seven': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}

ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)

tokens_to_ids = {}
ids_to_tokens = []
pad_index = None
empty_index = None

key_profile = pickle.load(open('getmusic/utils/key_profile.pickle', 'rb'))

pos_in_bar = beat_note_factor * max_notes_per_bar * pos_resolution

chord_pitch_out_of_key_prob = 0.01
key_change_prob = 0.001
chord_change_prob = 0.5
key_chord_distribution = _key_chord_distribution(
    chord_pitch_out_of_key_prob=chord_pitch_out_of_key_prob)
key_chord_loglik = np.log(key_chord_distribution)
key_chord_transition_distribution = _key_chord_transition_distribution(
    key_chord_distribution,
    key_change_prob=key_change_prob,
    chord_change_prob=chord_change_prob)
key_chord_transition_loglik = np.log(key_chord_transition_distribution)


class Item(object):
    def __init__(self, name, start, end, vel=0, pitch=0, track=0, value=''):
        self.name = name
        self.start = start  # start step
        self.end = end  # end step
        self.vel = vel
        self.pitch = pitch
        self.track = track
        self.value = value

    def __repr__(self):
        return f'Item(name={self.name:>10s}, start={self.start:>4d}, end={self.end:>4d}, ' \
               f'vel={self.vel:>3d}, pitch={self.pitch:>3d}, track={self.track:>2d}, ' \
               f'value={self.value:>10s})\n'

    def __eq__(self, other):
        return self.name == other.name and self.start == other.start and \
            self.pitch == other.pitch and self.track == other.track


def get_args():
    parser = argparse.ArgumentParser(description='sampling script')
    parser.add_argument('--config_file', type=str, default='configs/train.yaml',
                        help='path of config file')
    parser.add_argument('--name', type=str, default='inference_cache',
                        help='the name of this experiment, if not provided, set to'
                             'the name of config file')
    parser.add_argument('--output', type=str, default='cache',
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')
    parser.add_argument('--num_node', type=int, default=NUM_NODE,
                        help='number of nodes for distributed training')
    parser.add_argument('--ngpus_per_node', type=int, default=8,
                        help='number of gpu on one node')
    parser.add_argument('--node_rank', type=int, default=NODE_RANK,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default=DIST_URL,
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--sync_bn', action='store_true',
                        help='use sync BN layer')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', action='store_true',
                        help='set cudnn.deterministic True')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='automatic mixture of precesion')
    parser.add_argument('--do_sample', action='store_false', default=True)
    parser.add_argument('--conditional_name', type=str, default="c",
                        help="Specify the conditional track. " + \
                        "For example, if you want to generate based on the lead and chord tracks -> lc " + \
                        "('b' for bass, 'd' for drums, 'g' for guitar, 'l' for lead, 'p' for piano, 's' for strings, 'c' for chords)")
    parser.add_argument('--content_name', type=str, default="l",
                        help="Specify the content track. " + \
                        "For example, if you want to generate drum, guitar, and piano tracks -> dgp " + \
                        "('b' for bass, 'd' for drums, 'g' for guitar, 'l' for lead, 'p' for piano, 's' for strings, 'c' for chords)")
    parser.add_argument('--file_path', type=str, default=None)
    parser.add_argument('--skip_step', type=int, default=0)
    parser.add_argument('--decode_chord', action='store_true', default=False)
    parser.add_argument('--chord_from_single', action='store_true', default=False)
    parser.add_argument('--no_ema', action='store_false', default=True)

    parser.add_argument('--output_dir', type=str, default='./results',
                        help='directory to save the melody')

    parser.add_argument('--gen_seed', type=int, default=None,
                        help='seed for generation. int. (example) 0')

    parser.add_argument('--output_synth_demo', action="store_true", default=False)

    parser.add_argument('--one_shot_generation', action="store_true", default=False)
    parser.add_argument('--use_chroma_viterbi', action="store_true", default=False)

    parser.add_argument('--generate_from_calculated_chords', action="store_true", default=False)
    parser.add_argument('--sixteenth_times_and_countings_filepath', type=str, default=None)
    parser.add_argument('--conditional_chords_filepath', type=str, default=None)

    parser.add_argument('--gen_melody_num', type=int, default=2)

    parser.add_argument('--bpm', type=int, default=None)
    parser.add_argument('--start_time', type=float, default=None)

    parser.add_argument('--load_path', type=str, default=None,
                    help='path to model that need to be loaded, '
                            'used for loading pretrained model')

    parser.add_argument('--bgm_filepath', type=str, required=True)


    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    args.cwd = os.path.abspath(os.path.dirname(__file__))

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.name == '':
        args.name = os.path.basename(args.config_file).replace('.yaml', '')

    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    now = (datetime.datetime.now() - random_seconds_shift).strftime('%Y-%m-%dT%H-%M-%S')
    args.save_dir = os.path.join(args.output, args.name, now)

    if args.gen_seed is None:
        args.gen_seed = random.randint(MIN_GEN_SEED, MAX_GEN_SEED)
    else:
        assert MIN_GEN_SEED <= args.gen_seed <= MAX_GEN_SEED

    if not os.path.exists(args.bgm_filepath):
        assert False, "The specified bgm filepath does not exist."
    filename, file_extension = os.path.splitext(os.path.basename(args.bgm_filepath))
    args.filename = filename
    args.file_extension = file_extension

    if args.generate_from_calculated_chords:
        assert args.sixteenth_times_and_countings_filepath is not None and os.path.exists(args.sixteenth_times_and_countings_filepath), \
            "if you want to generete from calculated chords, you hove to specify right sixteenth_times_and_countings_filepath"
        assert os.path.splitext(os.path.basename(args.sixteenth_times_and_countings_filepath))[1] in JSON_FILENAME_EXTENSIONS, \
            f"Supported extensions for sixteenth_times_and_countings_filepath are: {', '.join(JSON_FILENAME_EXTENSIONS)}"
        assert args.conditional_chords_filepath is not None and os.path.exists(args.conditional_chords_filepath), \
            "if you want to generete from calculated chords, you hove to specify right conditional_chords_filepath"
        assert os.path.splitext(os.path.basename(args.conditional_chords_filepath))[1] in MIDI_FILENAME_EXTENSIONS, \
            f"Supported extensions for conditional_chords_filepath are: {', '.join(MIDI_FILENAME_EXTENSIONS)}"
        if args.output_synth_demo:
            assert file_extension in AUDIO_FILENAME_EXTENSIONS, \
                "if you want to generete from calculated chords and want to make synth demo, you hove to specify original audio file as bgm file"
        args.from_audio = True
    else:
        if file_extension in AUDIO_FILENAME_EXTENSIONS:
            args.from_audio = True
        elif file_extension in MIDI_FILENAME_EXTENSIONS:
            args.from_audio = False
        else:
            assert False, f"Unsupported file extension: {file_extension}. " + \
                    f"Supported extensions are: {', '.join(AUDIO_FILENAME_EXTENSIONS + MIDI_FILENAME_EXTENSIONS)}"

    if (args.bpm is not None and args.start_time is None) or (args.start_time is not None and args.bpm is None):
        assert False, "If either --bpm or --start_time is specified, the other must also be specified."
    elif args.bpm is not None and args.start_time is not None:
        args.use_handinputed_bpm = True
    else:
        args.use_handinputed_bpm = False

    return args


def normalize_to_c_major(e):
    def get_pitch_class_histogram(notes, use_duration=True, use_velocity=True, normalize=True):
        weights = np.ones(len(notes))
        if use_duration:
            weights *= [note[4] for note in notes]
        if use_velocity:
            weights *= [note[5] for note in notes]
        histogram, _ = np.histogram([note[3] % 12 for note in notes], bins=np.arange(
            13), weights=weights, density=normalize)
        if normalize:
            histogram /= (histogram.sum() + (histogram.sum() == 0))
        return histogram

    pitch_histogram = [i for i in e if i[2] < 128]
    if len(pitch_histogram) == 0:
        return e, True, 0

    histogram = get_pitch_class_histogram(pitch_histogram)
    key_candidate = np.dot(key_profile, histogram)
    key_temp = np.where(key_candidate == max(key_candidate))
    major_index = key_temp[0][0]
    minor_index = key_temp[0][1]
    major_count = histogram[major_index]
    minor_count = histogram[minor_index % 12]
    key_number = 0
    if major_count < minor_count:
        key_number = minor_index
        is_major = False
    else:
        key_number = major_index
        is_major = True
    real_key = key_number
    # transposite to C major or A minor
    if real_key <= 11:
        trans = 0 - real_key
    else:
        trans = 21 - real_key
    pitch_shift = trans

    e = [tuple(k + pitch_shift if j == 3 and i[2] != 128 else k for j, k in enumerate(i))
         for i in e]
    return e, is_major, pitch_shift

def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]

def e2t(x):
    return ts_list[x]

def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]

def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]

def v2e(x):
    return x // velocity_quant

def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)

def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e

def e2b(x):
    return math.floor(2 ** (x / tempo_quant) * min_tempo)

def time_signature_reduce(numerator, denominator):
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator

def MIDI_to_encoding(midi_obj, with_chord, condition_inst, chord_from_single):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    max_pos = max(notes_start_pos) + 1

    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []

    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue

            info = pos_to_info[time_to_pos(note.start)]
            duration = d2e(time_to_pos(note.end) - time_to_pos(note.start))
            encoding.append([info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                            1 if inst.is_drum else note.pitch, duration, v2e(note.velocity), info[1], info[3]])
    if len(encoding) == 0:
        return list()

    encoding.sort()
    encoding, is_major, pitch_shift = normalize_to_c_major(encoding)

    # extract chords
    if with_chord:
        max_pos = 0
        note_items = []
        for note in encoding:
            if (0 < note[3] < 128) and (note[2] in [0,25,32,48,80]):
                if chord_from_single and (str(note[2]) not in condition_inst):
                    continue

                ts = e2t(note[6])
                measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
                max_pos = max(
                    max_pos, measure_length * note[0] + note[1] + e2d(note[4]))
                note_items.append(Item(
                    name='On',
                    start = measure_length * note[0] + note[1],
                    end = measure_length * note[0] + note[1] + e2d(note[4]),
                    vel=e2v(note[5]),
                    pitch=note[3],
                    track=0))
        note_items.sort(key=lambda x: (x.start, -x.end))
        pos_per_chord = measure_length
        # pos_per_chord = measure_length // ts[0] * 2
        max_chords = round(max_pos // pos_per_chord + 0.5)
        if max_chords > 0:
            chords = infer_chords_for_sequence(note_items,
                                        pos_per_chord=pos_per_chord,
                                        max_chords=max_chords,
                                        key_chord_loglik=key_chord_loglik,
                                        key_chord_transition_loglik=key_chord_transition_loglik,
                                        pitch_shift=pitch_shift)
        else:
            chords = []

        bar_idx = 0
        for c in chords:
            if c == 'N.C.':
                bar_idx+=1
                continue
            r, k = c.split(':')
            if k == '':
                k = 'null'
            elif k == '7':
                k = 'seven'
            encoding.append((bar_idx, 0, 129, root_dict[r], kind_dict[k], 0, t2e(time_signature_reduce(4, 4)), 0))
            bar_idx += 1

        encoding.sort()

    return encoding, pitch_shift, tpc

def encoding_to_MIDI(encoding, tpc, decode_chord):

    tmp = encoding.strip().split('<0-')[1:]

    encoding = []
    for item in tmp:
        tmp2 = item.strip()[:-1].split('> <')
        encoding.append([int(tmp2[0])] + [int(i[2:]) for i in tmp2[1:]])
    del tmp

    bar_to_timesig = [list()
                      for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(
        i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = t2e(time_signature_reduce(
                4, 4)) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = e2t(bar_to_timesig[i])
        measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]

    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.tempo_changes = tpc

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]

    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]

        if program == 129 and decode_chord:
            root_name = root_list[i[3]]
            kind_name = kind_list[i[4]]
            root_pitch_shift = root_dict[root_name]
            end = start + get_tick(0, e2d(1))
            for kind_shift in _CHORD_KIND_PITCHES[kind_name]:
                pitch = 36 + root_pitch_shift + kind_shift
                midi_obj.instruments[1].notes.append(miditoolkit.containers.Note(
                start=start, end=end, pitch=pitch, velocity=e2v(20)))
        elif program != 129:
            pitch = (i[3] - 128 if program == 128 else i[3])
            if pitch < 0:
                continue
            duration = get_tick(0, e2d(i[4]))
            if duration == 0:
                duration = 1
            end = start + duration
            velocity = e2v(i[5])

            midi_obj.instruments[program].notes.append(miditoolkit.containers.Note(
                start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = e2t(new_ts)
            midi_obj.time_signature_changes.append(miditoolkit.containers.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj

def F(file_name, conditional_tracks, content_tracks, condition_inst, chord_from_single):

    global tokens_to_ids
    global ids_to_tokens
    global empty_index
    global pad_index

    empty_tracks = ~conditional_tracks & ~content_tracks

    conditional_tracks &= ~empty_tracks # emptied tracks can not be condition
    conditional_tracks = torch.tensor(conditional_tracks).float()
    conditional_tracks = conditional_tracks.view(7,1).repeat(1,2).reshape(14,1)
    empty_tracks = torch.tensor(empty_tracks).float()
    empty_tracks = empty_tracks.view(7,1).repeat(1,2).reshape(14,1)

    midi_obj = miditoolkit.midi.parser.MidiFile(file_name)

    if conditional_tracks[-1]:
        with_chord = True
    else:
        with_chord = False

    # try:
    encoding, pitch_shift, tpc = MIDI_to_encoding(midi_obj, with_chord, condition_inst, chord_from_single)

    if len(encoding) == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None, 0

    bar_index_offset = 0

    figure_size = encoding[-1][0] * pos_in_bar + encoding[-1][1]

    pad_length = 1 #(512 - figure_size % 512)

    figure_size += pad_length

    conditional_bool = conditional_tracks.repeat(1,figure_size)

    empty_pos = empty_tracks.repeat(1, figure_size).type(torch.bool)

    datum = pad_index * torch.ones(14, figure_size, dtype=float)

    oov = 0
    inv = 0

    chord_list = []

    tempo = b2e(67)

    lead_start = 0

    idx = 0
    while idx != len(encoding) - 1:
        e = encoding[idx]

        bar = e[0]
        pos = e[1]
        inst = e[2]
        pitch = e[3]

        if inst == 80:
            tempo = e[7]
            assert tempo != 0, 'bad tempo'

        # assert e[6] == 6

        if e[2] == 129:
            row = inst_to_row[str(inst)]
            r = root_list[e[3]]
            k = kind_list[e[4]]
            datum[2 * row][pos_in_bar * bar + pos : pos_in_bar * (bar + 1) + pos] = tokens_to_ids[r]
            datum[2 * row + 1][pos_in_bar * bar + pos : pos_in_bar * (bar + 1) + pos] = tokens_to_ids[k]
            idx += 1
            continue

        chord_list = [str(e[3])]

        for f_idx in range(idx + 1, len(encoding)):
            if (encoding[f_idx][0] == bar) and (encoding[f_idx][1] == pos) and (encoding[f_idx][2] == inst):
                if encoding[f_idx][3] != pitch:
                    chord_list.append(str(encoding[f_idx][3]))
                    pitch = encoding[f_idx][3]
            else:
                break

        idx = max(idx + 1, f_idx)


        dur = e[4]
        if dur == 0:
            continue

        if not (str(inst) in inst_to_row):
            continue

        row = inst_to_row[str(inst)]
        dur = tokens_to_ids['T'+str(e[4])] # duration

        chord_string = ' '.join(chord_list)
        token = prog_to_abrv[str(inst)] + chord_string

        if token in tokens_to_ids:
            pitch = tokens_to_ids[token]
            assert (dur < pad_index) and (pitch > pad_index), 'pitch index is {} and dur index is {}'.format(pitch, dur)
            datum[2 * row][pos_in_bar * bar + pos] = pitch
            datum[2 * row + 1][pos_in_bar * bar + pos] = dur
            inv += 1
        else:
            oov += 1

    datum = torch.where(empty_pos, empty_index, datum)
    datum = torch.where(((datum != empty_index).float() * (1 - conditional_bool)).type(torch.bool), empty_index + 1, datum)

    # datum = datum[:,:512]
    # conditional_bool = conditional_bool[:,:512]

    # if trunc:
    # datum = datum[:,:512]
    # conditional_bool = conditional_bool[:,:512]

    not_empty_pos = (torch.tensor(np.array(datum)) != empty_index).float()

    have_cond = True

    for i in range(14):
        if with_chord and conditional_tracks[i] == 1 and ((datum[i] == pad_index).sum() + (datum[i] == empty_index).sum()) == min(512,figure_size):
            have_cond = False
            break

    return datum.unsqueeze(0), torch.tensor(tempo), not_empty_pos, conditional_bool, pitch_shift, tpc, have_cond

def load_json(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data

def aggregate_tracks(midi):
    # 参照 http://www.synapse.ne.jp/seiji/music/midi/gm.html https://ja.wikipedia.org/wiki/General_MIDI

    group_to_pnum_dict = {
        "piano": list(range(0, 8)),
        "chromatic_percussion": list(range(8, 16)),
        "organ": list(range(16, 24)),
        "guitar": list(range(24, 32)),
        "bass": list(range(32, 40)),
        "strings": list(range(40, 48)),
        "ensemble": list(range(48, 56)),
        "brass": list(range(56, 64)),
        "reed": list(range(64, 72)),
        "pipe": list(range(72, 80)),
        "synth_lead": list(range(80, 88)),
        "synth_pad": list(range(88, 96)),
        "synth_effects": list(range(96, 104)),
        "ethnic": list(range(104, 112)),
        "percussive": list(range(112, 120)),
        "sound_effects": list(range(120, 128))
    }

    lead = Instrument(program=80, is_drum=False, name="lead")
    bass = Instrument(program=32, is_drum=False, name="bass")
    drum = Instrument(program=0, is_drum=True, name="drum")
    guitar = Instrument(program=25, is_drum=False, name="guitar")
    piano = Instrument(program=0, is_drum=False, name="piano")
    string = Instrument(program=48, is_drum=False, name="string")

    for instrument in midi.instruments:
        program = instrument.program
        is_drum = instrument.is_drum

        if is_drum == True:
            drum.notes += instrument.notes

        else:
            # piano
            # ピアノ系 + クロマチック・パーカッション系 + オルガン系 + シンセ・パッド系
            if program in group_to_pnum_dict["piano"] + group_to_pnum_dict["chromatic_percussion"] + group_to_pnum_dict["organ"] + group_to_pnum_dict["synth_pad"]:
                piano.notes += instrument.notes

            # guitar
            # ギター系 + エスニック系
            elif program in group_to_pnum_dict["guitar"] + group_to_pnum_dict["ethnic"]:
                guitar.notes += instrument.notes

            # bass
            # ベース系
            elif program in group_to_pnum_dict["bass"]:
                bass.notes += instrument.notes

            # string
            # ストリングス系 + アンサンブル系 + ブラス系
            elif program in group_to_pnum_dict["strings"] + group_to_pnum_dict["ensemble"] + group_to_pnum_dict["brass"]:
                string.notes += instrument.notes

            # lead
            # シンセ・リード系 + リード系 +　パイプ系
            elif program in group_to_pnum_dict["synth_lead"] + group_to_pnum_dict["reed"] + group_to_pnum_dict["pipe"]:
                lead.notes += instrument.notes

            # others -> トラック削除
            # シンセ・エフェクト系 + パーカッシヴ系 + 効果音
            else:
                pass

    midi.instruments = []

    if len(lead.notes) != 0:
        midi.instruments.append(lead)

    if len(bass.notes) != 0:
        midi.instruments.append(bass)

    if len(drum.notes) != 0:
        midi.instruments.append(drum)

    if len(guitar.notes) != 0:
        midi.instruments.append(guitar)

    if len(piano.notes) != 0:
        midi.instruments.append(piano)

    if len(string.notes) != 0:
        midi.instruments.append(string)

    return midi

def convert_quantized_time_to_real_time(midi, clipped_sixteenth_times_and_countings):
    midi_real_time = copy.deepcopy(midi)
    sixteenth_unit_tick = midi_real_time.resolution / 4

    targets = [i for i in range(len(midi_real_time.instruments))]
    for target in targets:
        for i, note in enumerate(midi_real_time.instruments[target].notes):

            note_start_tick = midi_real_time.time_to_tick(note.start)
            note_start_idx = min(round(note_start_tick / sixteenth_unit_tick), len(clipped_sixteenth_times_and_countings) - 1)
            note_start_real_time = clipped_sixteenth_times_and_countings[note_start_idx][0]

            note_end_tick = midi_real_time.time_to_tick(note.end)
            note_end_idx = min(round(note_end_tick / sixteenth_unit_tick), len(clipped_sixteenth_times_and_countings) - 1)
            note_end_real_time = clipped_sixteenth_times_and_countings[note_end_idx][0]

            midi_real_time.instruments[target].notes[i].start = note_start_real_time
            midi_real_time.instruments[target].notes[i].end = note_end_real_time

    midi_real_time.remove_invalid_notes()

    return midi_real_time

def compose_dynamic_tempo_midi(midi, clipped_sixteenth_times_and_countings):

    # define const
    SEC_PER_MIN = 60
    BEATS_PER_BAR = 4
    SIXTEENTH_PER_BEAT = 4

    # calclulate offset ticks and sixteenth note's tick
    offset_tick = int(BEATS_PER_BAR * midi.resolution)
    sixteenth_tick = midi.resolution / SIXTEENTH_PER_BEAT

    # init
    dynamic_tempo_midi = copy.deepcopy(midi)

    # add offset bar
    offset_time = clipped_sixteenth_times_and_countings[0][0]
    offset_bar_bpm = SEC_PER_MIN * BEATS_PER_BAR / offset_time

    # prepare tick scales
    dynamic_tempo_midi._tick_scales = []
    tempo_changes = [(0, offset_bar_bpm)]

    # add tick scales according to sixteenth times and countings
    for i in range(len(clipped_sixteenth_times_and_countings) - 1):
        sixteenth_start_time = clipped_sixteenth_times_and_countings[i][0]
        sixteenth_end_time = clipped_sixteenth_times_and_countings[i+1][0]

        start_tick = int(clipped_sixteenth_times_and_countings[i][1] * dynamic_tempo_midi.resolution) + offset_tick
        bpm = SEC_PER_MIN / ((sixteenth_end_time - sixteenth_start_time) * SIXTEENTH_PER_BEAT)
        tempo_changes.append((start_tick, bpm))

    # add to midi and update tick to time
    for tick, bpm in tempo_changes:
        dynamic_tempo_midi._tick_scales.append((tick, SEC_PER_MIN / bpm / midi.resolution))
    dynamic_tempo_midi._update_tick_to_time(0)

    # add note to midi and quantize according to new scale
    dynamic_tempo_midi.instruments[0].notes = []
    for note in midi.instruments[0].notes:
        note.start = dynamic_tempo_midi.tick_to_time(int(round(dynamic_tempo_midi.time_to_tick(note.start) / sixteenth_tick) * sixteenth_tick))
        note.end = dynamic_tempo_midi.tick_to_time(int(round(dynamic_tempo_midi.time_to_tick(note.end) / sixteenth_tick) * sixteenth_tick))
        dynamic_tempo_midi.instruments[0].notes.append(note)

    return dynamic_tempo_midi

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


def main():

    ## prepare
    args = get_args()

    torch.cuda.set_device(0)
    args.ngpus_per_node = 1
    args.world_size = 1

    args.local_rank = 0

    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)

    global tokens_to_ids
    global ids_to_tokens
    global empty_index
    global pad_index

    with open(config['solver']['vocab_path'],'r') as f:
        tokens = f.readlines()

        for id, token in enumerate(tokens):
            token, freq = token.strip().split('\t')
            tokens_to_ids[token] = id
            ids_to_tokens.append(token)
        pad_index = tokens_to_ids['<pad>']
        empty_index = len(ids_to_tokens)

    model = build_model(config, args)

    dataloader_info = None

    solver = Solver(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger, is_sample=True)

    assert args.load_path is not None
    solver.resume(path=args.load_path)


    # specify which track is conditional or content
    conditional_track = np.array([False, False, False, False, False, False, False])
    conditional_name = args.conditional_name
    condition_inst = []
    for c, info in c2info.items():
        if c in conditional_name:
            conditional_track[info["track_id"]] = True
            if info["add_to_conditional_inst"]:
                condition_inst.append(info["program_num"])

    content_track = np.array([False, False, False, False, False, False, False])
    content_name = args.content_name
    for c, info in c2info.items():
        if c in content_name:
            content_track[info["track_id"]] = True

    if all(conditional_track):
        print('You can\'t set all tracks as condition. We conduct uncontional generation based on selected content tracks.')
        conditional_track = np.array([False, False, False, False, False, False, False])
        if not any(content_track):
            assert False, 'No content tracks is selected. skip this song'


    # prepare conditonal midi
    if args.from_audio:
        if args.generate_from_calculated_chords:
            conditional_midi = CustomPrettyMIDI(midi_file=args.conditional_chords_filepath)
            clipped_sixteenth_times_and_countings = load_json(filepath=args.sixteenth_times_and_countings_filepath)
        else:
            if args.use_chroma_viterbi:
                conditional_midi, clipped_sixteenth_times_and_countings = \
                    make_midi_from_chroma(audio_file_path=args.bgm_filepath, use_handinputed_bpm=args.use_handinputed_bpm, start_time=args.start_time, bpm=args.bpm)
            else:
                conditional_midi, clipped_sixteenth_times_and_countings = \
                    make_midi_by_madmom_chord_recognition(audio_file_path=args.bgm_filepath, use_handinputed_bpm=args.use_handinputed_bpm, start_time=args.start_time, bpm=args.bpm)

        with open(os.path.join(args.output_dir, SIXTEENTH_TIMES_AND_COUNTINGS_OUTPUT_FILENAME), "w") as file:
            json.dump(clipped_sixteenth_times_and_countings, file)
        conditional_midi.write(filename=os.path.join(args.output_dir, CONDITONAL_CHORDS_OUTPUT_FILENAME))

    else:
        conditional_midi = CustomPrettyMIDI(midi_file=args.bgm_filepath)

    conditional_midi = aggregate_tracks(conditional_midi)
    conditional_midi.write(filename=TMP_MIDIFILE)
    file_name = TMP_MIDIFILE


    # generation
    x, tempo, not_empty_pos, condition_pos, pitch_shift, tpc, have_cond = F(file_name, conditional_track, content_track, condition_inst, args.chord_from_single)

    if not have_cond:
        assert False, "there exists the track that is specified as conditional inst but doesn't have any note in its track."

    if args.one_shot_generation:
        oct_lines = solver.infer_multi_sample(x, tempo, not_empty_pos, condition_pos,
                                       seed=args.gen_seed, cudnn_deterministic=args.cudnn_deterministic, use_ema=args.no_ema,
                                       gen_melody_num=args.gen_melody_num)
    else:
        oct_lines = solver.infer_multi_sample_by_split(x, tempo, not_empty_pos, condition_pos, split_size=INFER_SPLIT_SIZE,
                                    seed=args.gen_seed, cudnn_deterministic=args.cudnn_deterministic, use_ema=args.no_ema,
                                    gen_melody_num=args.gen_melody_num)

    for melody_idx in range(len(oct_lines)):
        os.makedirs(os.path.join(args.output_dir, str(melody_idx)), exist_ok=True)

        oct_line = oct_lines[melody_idx]

        data = oct_line.split(' ')

        oct_final_list = []
        for start in range(3, len(data),8):
            if 'pad' not in data[start] and 'pad' not in data[start+1]:
                pitch = int(data[start][:-1].split('-')[1])
                if data[start-1] != '<2-129>' and data[start-1] != '<2-128>':
                    pitch -= pitch_shift
                data[start] = '<3-{}>'.format(pitch) # re-normalize
                oct_final_list.append(' '.join(data[start-3:start+5]))

        oct_final = ' '.join(oct_final_list)

        midi_obj = encoding_to_MIDI(oct_final, tpc, args.decode_chord)


        # remove tracks except for lead track
        has_lead_track = False
        for instrument in midi_obj.instruments:
            if instrument.program == int(c2info["l"]["program_num"]):
                midi_obj.instruments = [instrument]
                has_lead_track = True
        if not has_lead_track:
            print("!generated track doesn't have lead track. So the following processes will be carried out with an empty track")
            midi_obj.instruments = [miditoolkit.midi.containers.Instrument(program=int(c2info["l"]["program_num"]), is_drum=False)]


        # notes arrangements
        midi_obj.tempo_changes = midi_obj.tempo_changes[:-1] # I don't know why, but TempoChange of wrong bpm forcibly added to MIDI, so remove it.
        midi_obj.dump(TMP_MIDIFILE)
        melody_midi = CustomPrettyMIDI(midi_file=TMP_MIDIFILE)
        melody_midi.perfect_monophonize(targets="all")
        melody_midi.note_shift(targets="all")
        melody_midi.remove_invalid_notes()
        if args.from_audio:
            melody_midi = \
                convert_quantized_time_to_real_time(midi=melody_midi, clipped_sixteenth_times_and_countings=clipped_sixteenth_times_and_countings)
            melody_midi = compose_dynamic_tempo_midi(midi=melody_midi, clipped_sixteenth_times_and_countings=clipped_sixteenth_times_and_countings)
        melody_midi.write(filename=os.path.join(args.output_dir, str(melody_idx), MELODY_MIDI_OUTPUT_FILENAME))

        # mix
        if args.output_synth_demo:
            if args.from_audio:
                bgm_audio, _ = librosa.load(args.bgm_filepath, sr=OUTPUT_SR)
            else:
                bgm_midi = CustomPrettyMIDI(midi_file=args.bgm_filepath)
                bgm_audio = bgm_midi.fluidsynth(fs=OUTPUT_SR)

            melody_audio = melody_midi.fluidsynth(fs=OUTPUT_SR)
            mixed_audio = mix_audio(bgm_audio, melody_audio, weight2=MELODY_WEIGHT)

            sf.write(os.path.join(args.output_dir, str(melody_idx), MELODY_AUDIO_OUTPUT_FILENAME), melody_audio, OUTPUT_SR)
            sf.write(os.path.join(args.output_dir, str(melody_idx), MIXED_AUDIO_OUTPUT_FILENAME), mixed_audio, OUTPUT_SR)


if __name__ == '__main__':
    main()
