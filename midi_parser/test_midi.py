import os
import torch

from midi_parser.MidiData import MidiData


def preprocess_midi_files(midi_files_folder, device, num_of_samples=100, num_of_notes=20):
    files = sorted([midi_files_folder + "/" + file for file in os.listdir(midi_files_folder)])

    notes_cpu = []
    notes_start_cpu = []
    notes_length_cpu = []
    attacks_cpu = []
    decays_cpu = []
    for file in files:
        _notes, _notes_length, _notes_start, _attacks = test_midi_data_custom(file)
        _decays = [0.1] * len(_notes)
        notes_cpu.append(_notes)
        notes_start_cpu.append(_notes_start)
        notes_length_cpu.append(_notes_length)
        attacks_cpu.append(_attacks)
        decays_cpu.append(_decays)
            
    notes, notes_start, notes_length, attacks, decays = preprocess_midi_data(
        device,
        notes_cpu,
        notes_start_cpu,
        notes_length_cpu,
        attacks_cpu,
        decays_cpu,
        num_of_samples,
        num_of_notes
    )
    attacks = attacks * 0.2

    return notes, notes_start, notes_length, attacks, decays

def test_midi_data_custom(midi_file):
    midi_data = MidiData(midi_file)
    notes = []
    note_lengths = []
    start_times = []
    attacks = []
    for i in range(midi_data.get_num_tracks()):
        track = midi_data.get_track(i)
        if i == 0:
            continue
        for note in track.notes:
            start_time = round(note.start_time * .001, 2)
            note_length = round(note.end_time * .001 - note.start_time * .001, 2)
            note_velocity = round((note.velocity / 127) * note_length, 2) 
            notes.append(note.pitch)
            note_lengths.append(note_length)
            start_times.append(start_time)
            attacks.append(note_velocity)
    order = sorted(range(len(start_times)), key=lambda i: start_times[i])
    start_times = [start_times[i] for i in order]
    notes = [notes[i] for i in order]
    note_lengths = [note_lengths[i] for i in order]
    attacks = [attacks[i] for i in order]
    
    return notes, note_lengths, start_times, attacks

def preprocess_midi_data(device, notes, notes_start, notes_length, attacks, decays, num_of_samples=100, num_of_notes=10):
    max_len = max(len(arr) for arr in notes)
    padded_notes = [(arr + [0] * (max_len - len(arr)))[:num_of_notes] for arr in notes][:num_of_samples]
    padded_notes_start = [(arr + [0] * (max_len - len(arr)))[:num_of_notes] for arr in notes_start][:num_of_samples]
    padded_notes_length = [(arr + [0] * (max_len - len(arr)))[:num_of_notes] for arr in notes_length][:num_of_samples]
    padded_attacks = [(arr + [0] * (max_len - len(arr)))[:num_of_notes] for arr in attacks][:num_of_samples]
    padded_decays = [(arr + [0] * (max_len - len(arr)))[:num_of_notes] for arr in decays][:num_of_samples]
    notes = torch.tensor(padded_notes, device=device)
    notes_start = torch.tensor(padded_notes_start, device=device)
    notes_length = torch.tensor(padded_notes_length, device=device)
    attacks = torch.tensor(padded_attacks, device=device)
    decays = torch.tensor(padded_decays, device=device)
    return notes, notes_start, notes_length, attacks, decays

