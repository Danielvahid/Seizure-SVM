"""Build a cached feature matrix from raw CHB-MIT EDF files (run once, slow)."""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eegsvm.data import load_recording, make_windows, parse_summary
from eegsvm.features import extract_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--patients", nargs="+", default=[f"chb{i:02d}" for i in range(1, 11)])
    p.add_argument("--neg-frac", type=float, default=0.1,
                   help="fraction of seizure-free windows to keep (all seizure windows are kept)")
    p.add_argument("--out", default="data/features.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    X_all, y_all, g_all = [], [], []

    for patient in args.patients:
        pdir = Path(args.data_dir) / patient
        seizures = parse_summary(pdir / f"{patient}-summary.txt")
        for edf in tqdm(sorted(pdir.glob("*.edf")), desc=patient):
            raw = load_recording(edf)
            if raw is None:
                tqdm.write(f"skipping {edf.name}: montage mismatch")
                continue
            windows, y = make_windows(raw, seizures.get(edf.name, []))
            # random subsampling of negatives keeps sensitivity/specificity/AUC unbiased
            keep = (y == 1) | (rng.random(len(y)) < args.neg_frac)
            X_all.append(extract_features(windows[keep], raw.info["sfreq"]))
            y_all.append(y[keep])
            g_all.append(np.full(keep.sum(), patient))

    X, y, groups = np.vstack(X_all), np.concatenate(y_all), np.concatenate(g_all)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, groups=groups)

    print(f"saved {args.out}: {X.shape[0]} windows, {X.shape[1]} features, {y.sum()} seizure windows")
    for patient in args.patients:
        m = groups == patient
        print(f"  {patient}: {m.sum()} windows, {y[m].sum()} seizure")


if __name__ == "__main__":
    main()
