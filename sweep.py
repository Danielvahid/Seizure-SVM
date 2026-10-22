"""C / gamma sweep for the RBF SVM, cross-validated across patients (GroupKFold)."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features.npz")
    p.add_argument("--max-neg", type=int, default=8000, help="cap on seizure-free windows to keep the sweep fast")
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    d = np.load(args.features)
    X, y, groups = d["X"], d["y"], d["groups"]

    rng = np.random.default_rng(0)
    neg = np.flatnonzero(y == 0)
    keep = np.concatenate([
        np.flatnonzero(y == 1),
        rng.choice(neg, min(len(neg), args.max_neg), replace=False),
    ])

    grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [1e-4, 1e-3, 1e-2, 1e-1]}
    pipe = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
    cv = GroupKFold(n_splits=min(5, len(set(groups[keep]))))
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=args.n_jobs)
    gs.fit(X[keep], y[keep], groups=groups[keep])
    print("best:", gs.best_params_, f"AUC={gs.best_score_:.3f}")

    res = pd.DataFrame(gs.cv_results_)
    res["C"] = res["param_svc__C"].astype(float)
    res["gamma"] = res["param_svc__gamma"].astype(float)
    table = res.pivot(index="C", columns="gamma", values="mean_test_score")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(table.values, cmap="viridis")
    ax.set_xticks(range(table.shape[1]), labels=[f"{g:g}" for g in table.columns])
    ax.set_yticks(range(table.shape[0]), labels=[f"{c:g}" for c in table.index])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title("Patient-grouped CV ROC AUC")
    for (i, j), v in np.ndenumerate(table.values):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w")
    fig.colorbar(im)
    fig.tight_layout()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    fig.savefig(out / "svm_sweep.png", dpi=150)


if __name__ == "__main__":
    main()
