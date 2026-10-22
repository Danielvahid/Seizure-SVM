# Patient-Independent Seizure Detection with SVMs

Detecting epileptic seizures from scalp EEG using Support Vector Machines, evaluated with **leave-one-patient-out** cross-validation, so every model is tested on a patient it has never seen.

## Why this matters

Most published seizure detectors are tuned per patient. A clinically useful device must generalise to new patients without retraining. This project measures how well an SVM does exactly that, and compares it against logistic regression, a linear SVM and a random forest.

## Dataset

[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (PhysioNet): continuous recordings from 23 paediatric patients with intractable seizures, sampled at 256 Hz. Seizure onsets and offsets are annotated in each patient's `chbXX-summary.txt`.

By default this project uses patients `chb01`-`chb10` (the full dataset is over 40 GB):

```bash
for p in chb01 chb02 chb03 chb04 chb05 chb06 chb07 chb08 chb09 chb10; do
  wget -r -N -c -np -nH --cut-dirs=3 -P data/raw https://physionet.org/files/chbmit/1.0.0/$p/
done
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python extract_features.py            # EDF -> data/features.npz (slow, run once)
python sweep.py                       # optional: C / gamma heatmap
python evaluate.py --C 10 --gamma 0.01
```

`evaluate.py` writes per-patient metrics to `results/lopo_results.csv` and a boxplot to `results/auc_by_model.png`.

## Method

1. **Preprocessing**: 18-channel bipolar montage, 0.5-45 Hz band-pass filter.
2. **Windowing**: 4 s non-overlapping windows; a window is labelled seizure if at least 50% of it lies inside an annotated seizure.
3. **Features** (9 per channel, 162 total): relative power in the delta, theta, alpha, beta and gamma bands, log total power, Hjorth mobility and complexity, and log line length.
4. **Models**: RBF SVM (main), linear SVM, logistic regression, random forest. All use class weighting to handle imbalance, and the SVMs standardise features.
5. **Evaluation**: leave-one-patient-out, reporting ROC AUC, sensitivity, specificity and false alarms per hour.

## Evaluation notes

- Seizures are extremely rare, so seizure-free windows are randomly subsampled (`--neg-frac`) to keep training tractable. Random subsampling of negatives leaves sensitivity, specificity and AUC unbiased, so these are the reported metrics. Precision and F1 are not reported because they depend on class prevalence.
- False alarms per hour are derived from specificity: `(1 - specificity) x 900` windows per hour.
- Metrics are computed per window, not per seizure event.

## Results

Run `evaluate.py` to populate this table (mean ± std across held-out patients).

| Model | AUC | Sensitivity | Specificity | False alarms / h |
|---|---|---|---|---|
| RBF SVM | | | | |
| Linear SVM | | | | |
| Logistic regression | | | | |
| Random forest | | | | |

## Limitations

- `sweep.py` tunes C and gamma on all loaded patients, so the held-out scores in `evaluate.py` are slightly optimistic. For a strict estimate, nest the sweep inside each training fold.
- Only the common 18-channel montage is used; recordings that lack it are skipped.
- Window-level metrics do not capture detection latency or event-level performance.

## Structure

```
eegsvm/data.py        parsing, filtering, windowing
eegsvm/features.py    feature extraction
extract_features.py   build the cached feature matrix
sweep.py              C / gamma grid with patient-grouped CV
evaluate.py           leave-one-patient-out comparison
```
