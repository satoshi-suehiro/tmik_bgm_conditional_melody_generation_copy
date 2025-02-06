import os
import sys

import mido
import pretty_midi

import numpy as np



class CustomPrettyMIDI(pretty_midi.PrettyMIDI):
    def __init__(
        self,
        midi_file: str | None = None,
        resolution: int = 480, # default resolution in original pretty_midi is 220.
        initial_tempo: float = 120.
    ):
        super().__init__(
            midi_file = midi_file,
            resolution = resolution,
            initial_tempo = initial_tempo
        )


    def quantize(
        self,
        targets: str | list[int],
        quantization_unit: int = 16
    ):
        '''
        Parameters
        ----------
            targets : str | list[int]
                List of indices, which indicate id of target instrument for quantization in self.instruments.
                Each index must be less than length of self.instruments.
                If you want to quantize all instruments, you can set targets as "all".

            quantization_unit : int
                For example, if you want to quantize notes by 16th note unit, set quantization_unit as 16.
                Quantization_unit must be in [16, 8, 4].

                    16 : 16th note
                    8 : 8th note
                    4 : quarter note
        '''

        if isinstance(targets, str):
            assert targets == "all"
            targets = [i for i in range(len(self.instruments))]
        elif isinstance(targets, list):
            for target in targets:
                assert isinstance(target, int)
                assert target < len(self.instruments)
        else:
            assert False, "targets should be 'all' or <list[int]>"

        assert quantization_unit in [16, 8, 4]
        quantization_unit_ticks = int(self.resolution / (quantization_unit // 4))

        for target in targets:
            for i, note in enumerate(self.instruments[target].notes):
                start_tick = self.time_to_tick(note.start)
                end_tick = self.time_to_tick(note.end)

                qtzd_start_tick = round(start_tick / quantization_unit_ticks) * quantization_unit_ticks
                qtzd_end_tick = round(end_tick / quantization_unit_ticks) * quantization_unit_ticks

                qtzd_start = self.tick_to_time(qtzd_start_tick)
                qtzd_end = self.tick_to_time(qtzd_end_tick)

                self.instruments[target].notes[i].start = qtzd_start
                self.instruments[target].notes[i].end = qtzd_end


    def extend_zero_length_notes(
        self,
        targets: str | list[int],
        minimum_length: int = 16
    ):
        '''extend notes which have same start and end to have minimum length.

        Parameters
        ----------
            targets : str | list[int]
                List of indices, which indicate id of target instrument for extension in self.instruments.
                Each index must be less than length of self.instruments.
                If you want to quantize all instruments, you can set targets as "all".

            minimum_length : int
                For example, if you want to extend note to length of 16th note, set minimum_length as 16.
                minimum_length must be in [16, 8, 4].

                    16 : 16th note
                    8 : 8th note
                    4 : quarter note
        '''

        if isinstance(targets, str):
            assert targets == "all"
            targets = [i for i in range(len(self.instruments))]
        elif isinstance(targets, list):
            for target in targets:
                assert isinstance(target, int)
                assert target < len(self.instruments)
        else:
            assert False, "targets should be 'all' or <list[int]>"

        assert minimum_length in [16, 8, 4]
        minimum_length_ticks = int(self.resolution / (minimum_length // 4))

        for target in targets:
            for i, note in enumerate(self.instruments[target].notes):
                start_tick = self.time_to_tick(note.start)
                end_tick = self.time_to_tick(note.end)

                if start_tick == end_tick:
                    extended_end_tick = int(end_tick + minimum_length_ticks)
                    self.instruments[target].notes[i].end = self.tick_to_time(extended_end_tick)


    def perfect_monophonize(
        self,
        targets: str | list[int]
    ):
        '''monophonize tracks. 'monophonize' means 'make track not to have notes that played at a time.'

        Parameters
        ----------
            targets : str | list[int]
                List of indices, which indicate id of target instrument for 'monophonization' in self.instruments.
                Each index must be less than length of self.instruments.
                If you want to qmonophonize all instruments, you can set targets as "all".

        '''
        REST = -1

        if isinstance(targets, str):
            assert targets == "all"
            targets = [i for i in range(len(self.instruments))]
        elif isinstance(targets, list):
            for target in targets:
                assert isinstance(target, int)
                assert target < len(self.instruments)
        else:
            assert False, "targets should be 'all' or <list[int]>"


        for target in targets:
            end_time = self.instruments[target].get_end_time()

            # record to matrix to monophonize
            pitch_matrix = np.array([REST for _ in range(self.time_to_tick(end_time))], dtype="int")
            vel_matrix = np.array([REST for _ in range(self.time_to_tick(end_time))], dtype="int")
            id_matrix = np.array([REST for _ in range(self.time_to_tick(end_time))], dtype="int")

            for note_id, note in enumerate(self.instruments[target].notes):
                start_tick = self.time_to_tick(note.start)
                end_tick = self.time_to_tick(note.end)

                pitch_matrix[start_tick:end_tick] = note.pitch
                vel_matrix[start_tick:end_tick] = note.velocity
                id_matrix[start_tick:end_tick] = note_id


            # make new notes according to matrix
            new_notes = []
            on_note = False

            # process first note
            before_id = id_matrix[0]
            if before_id != REST:
                on_note = True
                new_notes.append(
                    pretty_midi.Note(
                        start=self.tick_to_time(0),
                        end=None,
                        pitch=None,
                        velocity=None
                    )
                )

            # process notes
            for tick in range(1, len(id_matrix)):
                id = id_matrix[tick]
                if on_note:
                    if id == REST:
                        on_note = False

                        new_notes[-1].end = self.tick_to_time(tick)
                        new_notes[-1].pitch = pitch_matrix[tick-1]
                        new_notes[-1].velocity = vel_matrix[tick-1]

                    else:
                        if id != before_id:
                            new_notes[-1].end = self.tick_to_time(tick)
                            new_notes[-1].pitch = pitch_matrix[tick-1]
                            new_notes[-1].velocity = vel_matrix[tick-1]

                            new_notes.append(
                                pretty_midi.Note(
                                    start=self.tick_to_time(tick),
                                    end=None,
                                    pitch=None,
                                    velocity=None
                                )
                            )

                else:
                    if id != REST:
                        on_note = True

                        new_notes.append(
                            pretty_midi.Note(
                                start=self.tick_to_time(tick),
                                end=None,
                                pitch=None,
                                velocity=None
                            )
                        )

                before_id = id

            #  process last note
            if new_notes[-1].end is None:
                new_notes[-1].end = end_time
                new_notes[-1].pitch = pitch_matrix[tick]
                new_notes[-1].velocity = vel_matrix[tick]


            self.instruments[target].notes = new_notes



    def whole_shift(
        self,
        targets: str | list[int],
        sex: str = "female",
        pitch_range_kind: str = "normal"
    ):
        '''shift the pitch of whole notes to fit the voice range. 
        The pitch name (e.g. C#) doesn't change, only the octave changes.
        There is no guarantee that all notes will be within pitch_range after the shift.

        Parameters
        ----------
            targets : str | list[int]
                List of indices, which indicate id of target instrument for 'monophonization' in self.instruments.
                Each index must be less than length of self.instruments.
                If you want to qmonophonize all instruments, you can set targets as "all".

            sex : str
                Supposed singer's sex.
                For example, if you set 'sex' as 'female', the range of note pitches is shifted to fit the range of female voice.
                sex must be in ['male', 'female'].

            pitch_range_kind : str
                Kind of pitch range.
                For example, if you set 'pitch_range' as 'falsetto', the range of note pitches is shifted to fit the range of falsetto voice.
                pitch_range must be in ['normal', 'falsetto']
        '''
        SHIFT_PITCH_RANGE_MALE_NORMAL_MIN = 43 #G2
        SHIFT_PITCH_RANGE_MALE_NORMAL_MAX = 67 #G4

        SHIFT_PITCH_RANGE_MALE_FALSETTO_MIN = 52 #E3
        SHIFT_PITCH_RANGE_MALE_FALSETTO_MAX = 76 #E5

        SHIFT_PITCH_RANGE_FEMALE_NORMAL_MIN = 52 #E3
        SHIFT_PITCH_RANGE_FEMALE_NORMAL_MAX = 72 #C5

        SHIFT_PITCH_RANGE_FEMALE_FALSETTO_MIN = 60 #C4
        SHIFT_PITCH_RANGE_FEMALE_FALSETTO_MAX = 81 #A5

        if isinstance(targets, str):
            assert targets == "all"
            targets = [i for i in range(len(self.instruments))]
        elif isinstance(targets, list):
            for target in targets:
                assert isinstance(target, int)
                assert target < len(self.instruments)
        else:
            assert False, "targets should be 'all' or <list[int]>"

        assert sex in ["male", "female"]
        assert pitch_range_kind in ["normal", "falsetto"]

        min_pitch = eval(f"SHIFT_PITCH_RANGE_{sex.upper()}_{pitch_range_kind.upper()}_MIN")
        max_pitch = eval(f"SHIFT_PITCH_RANGE_{sex.upper()}_{pitch_range_kind.upper()}_MAX")
        mid_pitch = (min_pitch + max_pitch) // 2

        for target in targets:
            avg_pitch = 0
            for note in self.instruments[target].notes:
                pitch = note.pitch
                avg_pitch += pitch
            if len(self.instruments[target].notes) != 0:
                avg_pitch /= len(self.instruments[target].notes)

            difference_pitch_shift_list = []

            for pitch_shift in range(-2, 3):
                new_avg_pitch = avg_pitch + pitch_shift * 12
                difference = abs(mid_pitch - new_avg_pitch)
                difference_pitch_shift_list.append((difference, pitch_shift))

            difference_pitch_shift_list = sorted(difference_pitch_shift_list)
            first_candidate_difference, first_candidate_pitch_shift = difference_pitch_shift_list[0]

            chosen_pitch_shift = first_candidate_pitch_shift

            new_notes = []
            for note_id in range(len(self.instruments[target].notes)):
                self.instruments[target].notes[note_id].pitch += chosen_pitch_shift * 12



    def note_shift(
        self,
        targets: str | list[int],
        sex: str = "female",
        pitch_range_kind: str = "normal"
    ):
        '''shift the pitch of each notes to fit the voice range. 
        The pitch name (e.g. C#) doesn't change, only the octave changes.
        It is guaranteed that all notes are within pitch_range after the shift.

        Parameters
        ----------
            targets : str | list[int]
                List of indices, which indicate id of target instrument for 'monophonization' in self.instruments.
                Each index must be less than length of self.instruments.
                If you want to qmonophonize all instruments, you can set targets as "all".

            sex : str
                Supposed singer's sex.
                For example, if you set 'sex' as 'female', the range of note pitches is shifted to fit the range of female voice.
                sex must be in ['male', 'female'].

            pitch_range_kind : str
                Kind of pitch range.
                For example, if you set 'pitch_range' as 'falsetto', the range of note pitches is shifted to fit the range of falsetto voice.
                pitch_range must be in ['normal', 'falsetto']
        '''
        SHIFT_PITCH_RANGE_MALE_NORMAL_MIN = 43 #G2
        SHIFT_PITCH_RANGE_MALE_NORMAL_MAX = 67 #G4

        SHIFT_PITCH_RANGE_MALE_FALSETTO_MIN = 52 #E3
        SHIFT_PITCH_RANGE_MALE_FALSETTO_MAX = 76 #E5

        SHIFT_PITCH_RANGE_FEMALE_NORMAL_MIN = 52 #E3
        SHIFT_PITCH_RANGE_FEMALE_NORMAL_MAX = 72 #C5

        SHIFT_PITCH_RANGE_FEMALE_FALSETTO_MIN = 60 #C4
        SHIFT_PITCH_RANGE_FEMALE_FALSETTO_MAX = 81 #A5

        if isinstance(targets, str):
            assert targets == "all"
            targets = [i for i in range(len(self.instruments))]
        elif isinstance(targets, list):
            for target in targets:
                assert isinstance(target, int)
                assert target < len(self.instruments)
        else:
            assert False, "targets should be 'all' or <list[int]>"

        assert sex in ["male", "female"]
        assert pitch_range_kind in ["normal", "falsetto"]

        min_pitch = eval(f"SHIFT_PITCH_RANGE_{sex.upper()}_{pitch_range_kind.upper()}_MIN")
        max_pitch = eval(f"SHIFT_PITCH_RANGE_{sex.upper()}_{pitch_range_kind.upper()}_MAX")

        for target in targets:
            for note_id, note in enumerate(self.instruments[target].notes):
                pitch = note.pitch
                while pitch < min_pitch:
                    pitch += 12
                while pitch > max_pitch:
                    pitch -= 12
                note.pitch = pitch
                self.instruments[target].notes[note_id] = note


    def invert_leap_notes_until_convergence(
        self,
        targets: str | list[int],
        num_of_loop: int = 2
    ):
        '''invert the octave of a note whose leap is too great relative to the adjacent note.
        In a single process, we look at the notes in order, starting from the first note.
        we repeat the process 'num_of_loop' times. (So it's not necessarily strictly convergence.)

        Parameters
        ----------
            targets : str | list[int]
                List of indices, which indicate id of target instrument for 'monophonization' in self.instruments.
                Each index must be less than length of self.instruments.
                If you want to qmonophonize all instruments, you can set targets as "all".

            num_of_loop : int
                How many times to execute _invert_leap_notes().
                The larger it is, the more likely the end result will converge.(while the computational cost increases)

        '''
        if isinstance(targets, str):
            assert targets == "all"
            targets = [i for i in range(len(self.instruments))]
        elif isinstance(targets, list):
            for target in targets:
                assert isinstance(target, int)
                assert target < len(self.instruments)
        else:
            assert False, "targets should be 'all' or <list[int]>"

        count = 0
        while count <= num_of_loop:
            self._invert_leap_notes(targets=targets)
            count += 1


    def _invert_leap_notes(
        self,
        targets: str | list[int],
    ):
        def weighted_difference(note_pitch, before_note_pitch, before_note_vel, next_note_pitch, next_note_vel):
            return abs(((note_pitch - before_note_pitch) * before_note_vel + (note_pitch - next_note_pitch) * next_note_vel) / (before_note_vel + next_note_vel))

        for target in targets:

            notes = self.instruments[target].notes

            if len(notes) <= 2:
                return 0

            new_notes = []
            new_notes.append(notes[0])

            for id in range(1, len(notes) - 1):
                before_note_pitch = new_notes[id-1].pitch
                before_note_vel = new_notes[id-1].velocity

                note_pitch = notes[id].pitch

                next_note_pitch = notes[id+1].pitch
                next_note_vel = notes[id+1].velocity

                oc_up_note_pitch = note_pitch + 12
                oc_down_note_pitch = note_pitch - 12

                tmp_l = []
                tmp_l.append((weighted_difference(note_pitch, before_note_pitch, before_note_vel, next_note_pitch, next_note_vel), note_pitch))
                tmp_l.append((weighted_difference(oc_up_note_pitch, before_note_pitch, before_note_vel, next_note_pitch, next_note_vel), oc_up_note_pitch))
                tmp_l.append((weighted_difference(oc_down_note_pitch, before_note_pitch, before_note_vel, next_note_pitch, next_note_vel), oc_down_note_pitch))
                tmp_l = sorted(tmp_l)

                note = notes[id]
                note.pitch = tmp_l[0][1]
                new_notes.append(note)

            new_notes.append(notes[-1])

            self.instruments[target].notes = new_notes

