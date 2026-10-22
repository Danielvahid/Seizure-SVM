"""Leave-one-patient-out (LOPO) evaluation of the SVM against baseline classifiers.

Each patient is held out once, so every model is scored on a brain it never saw in training.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from eegsvm.data import WINDOW_S

WINDOWS_PER_HOUR = 3600 / WINDOW_S


def build_models(C, gamma):
    return {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, class_weight="balanced")),
        "svm_linear": make_pipeline(StandardScaler(), LinearSVC(dual=False, class_weight="balanced")),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=0
        ),
    }


def run_fold(X, y, groups, patient, C, gamma):
    test = groups == patient
    X_tr, y_tr, X_te, y_te = X[~test], y[~test], X[test], y[test]
    pos, neg = y_te == 1, y_te == 0

    rows = []
    for name, model in build_models(C, gamma).items():
        model.fit(X_tr, y_tr)
        if hasattr(model, "decision_function"):
            scores, threshold = model.decision_function(X_te), 0.0
        else:
            scores, threshold = model.predict_proba(X_te)[:, 1], 0.5
        pred = scores > threshold

        sensitivity = pred[pos].mean() if pos.any() else np.nan
        specificity = (~pred[neg]).mean() if neg.any() else np.nan
        auc = roc_auc_score(y_te, scores) if pos.any() and neg.any() else np.nan
        rows.append({
            "patient": patient,
            "model": name,
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_alarms_per_hour": (1 - specificity) * WINDOWS_PER_HOUR,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features.npz")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--gamma", default="scale", help="'scale' or a float")
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    gamma = args.gamma if args.gamma == "scale" else float(args.gamma)

    d = np.load(args.features)
    X, y, groups = d["X"], d["y"], d["groups"]

    folds = Parallel(n_jobs=args.n_jobs)(
        delayed(run_fold)(X, y, groups, patient, args.C, gamma) for patient in sorted(set(groups))
    )
    df = pd.DataFrame([row for fold in folds for row in fold])

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "lopo_results.csv", index=False)

    metrics = ["auc", "sensitivity", "specificity", "false_alarms_per_hour"]
    print(df.groupby("model")[metrics].agg(["mean", "std"]).round(3).to_string())

    models = list(df["model"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([df.loc[df.model == m, "auc"].dropna() for m in models])
    ax.set_xticklabels(models)
    ax.set_ylabel("ROC AUC (held-out patient)")
    ax.set_title("Leave-one-patient-out performance")
    fig.tight_layout()
    fig.savefig(out / "auc_by_model.png", dpi=150)


if __name__ == "__main__":
    main()
