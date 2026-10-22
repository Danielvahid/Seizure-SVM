"""Hand-crafted EEG features: relative band power, total power, Hjorth parameters, line length."""
import numpy as np
from scipy.signal import welch

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
EPS = 1e-12


def extract_features(X, sfreq):
    """Map windows (n_windows, n_channels, n_samples) in volts to a (n_windows, n_channels * 9) matrix."""
    X = X * 1e6  # volts -> microvolts so EPS is negligible next to the signal variance

    freqs, psd = welch(X, fs=sfreq, nperseg=int(sfreq), axis=-1)
    total = psd[..., (freqs >= 0.5) & (freqs < 45)].sum(-1)
    rel_power = np.stack(
        [psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) for lo, hi in BANDS.values()], axis=-1
    ) / (total[..., None] + EPS)

    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var0, var1, var2 = X.var(-1), d1.var(-1), d2.var(-1)
    mobility = np.sqrt(var1 / (var0 + EPS))
    complexity = np.sqrt(var2 / (var1 + EPS)) / (mobility + EPS)
    line_length = np.abs(d1).mean(-1)

    feats = np.concatenate(
        [
            rel_power,
            np.log10(total + EPS)[..., None],
            mobility[..., None],
            complexity[..., None],
            np.log10(line_length + EPS)[..., None],
        ],
        axis=-1,
    )
    return feats.reshape(len(feats), -1)
