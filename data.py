"""Parsing and windowing of CHB-MIT scalp EEG recordings."""
import re
from pathlib import Path

import mne
import numpy as np

WINDOW_S = 4            # window length in seconds
SEIZURE_OVERLAP = 0.5   # fraction of a window inside a seizure needed to label it positive

# 18-channel bipolar montage shared by most CHB-MIT recordings
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]


def parse_summary(summary_path):
    """Return {edf_filename: [(seizure_start_s, seizure_end_s), ...]} from a patient summary file."""
    seizures, current, start = {}, None, None
    for line in Path(summary_path).read_text().splitlines():
        if line.startswith("File Name:"):
            current = line.split(":", 1)[1].strip()
            seizures[current] = []
        # patient chb01 writes "Seizure Start Time", later patients "Seizure 1 Start Time"
        m = re.match(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", line)
        if m:
            start = int(m.group(1))
        m = re.match(r"Seizure\s*\d*\s*End Time:\s*(\d+)", line)
        if m:
            seizures[current].append((start, int(m.group(1))))
    return seizures


def load_recording(path):
    """Load an EDF file, keep the common montage and band-pass filter. Returns None if channels are missing."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    # MNE de-duplicates the repeated T8-P8 channel as T8-P8-0 / T8-P8-1
    if "T8-P8-0" in raw.ch_names:
        raw.rename_channels({"T8-P8-0": "T8-P8"})
    if set(CHANNELS) - set(raw.ch_names):
        return None
    raw.pick(CHANNELS).reorder_channels(CHANNELS)
    raw.filter(0.5, 45.0, verbose="ERROR")
    return raw


def make_windows(raw, seizures):
    """Cut a recording into non-overlapping windows and label them.

    Returns X of shape (n_windows, n_channels, n_samples) and binary labels y.
    """
    data = raw.get_data()
    n = int(WINDOW_S * raw.info["sfreq"])
    n_win = data.shape[1] // n
    X = data[:, : n_win * n].reshape(len(CHANNELS), n_win, n).transpose(1, 0, 2)

    starts = np.arange(n_win) * WINDOW_S
    y = np.zeros(n_win, dtype=int)
    for s, e in seizures:
        overlap = np.clip(np.minimum(starts + WINDOW_S, e) - np.maximum(starts, s), 0, None)
        y[overlap / WINDOW_S >= SEIZURE_OVERLAP] = 1
    return X, y
