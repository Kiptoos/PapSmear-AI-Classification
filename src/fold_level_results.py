"""
Supplementary Table S10 - fold-level results for every backbone (Reviewer 3, comment 2.6).
Uses the existing out-of-fold predictions only (no retraining).

    python src\\fold_level_results.py

Writes outputs/results/table_fold_level_accuracy.csv (one row per backbone, protocol,
repetition and fold) and outputs/results/table_fold_level_summary.csv (mean, SD, min and
max over the 25 folds of each backbone and protocol).
"""
import glob
import os

import numpy as np
import pandas as pd

PRED_DIR = os.path.join("outputs", "predictions")
OUT_DIR = os.path.join("outputs", "results")
CLASS_NAMES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
PROB_COLS = [f"prob_{c}" for c in CLASS_NAMES]


def macro_f1(y, p, k=5):
    f1 = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        prec = tp / max(np.sum(p == c), 1)
        rec = tp / max(np.sum(y == c), 1)
        f1.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1))


def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*_oof.csv")))
    if not files:
        raise SystemExit(f"No *_oof.csv files in {PRED_DIR}")
    rows = []
    for f in files:
        d = pd.read_csv(f)
        d["pred"] = d[PROB_COLS].to_numpy().argmax(1)
        for (arch, strat, rep, fold), g in d.groupby(["arch", "strategy", "repeat", "fold"]):
            y, p = g["true_label"].to_numpy(), g["pred"].to_numpy()
            rows.append({"Backbone": arch, "Protocol": strat, "Repetition": int(rep) + 1, "Fold": int(fold) + 1,
                         "Validation images": len(g), "Source images": g["group"].nunique(),
                         "Accuracy (%)": round(100 * float(np.mean(y == p)), 2),
                         "Macro-F1 (%)": round(100 * macro_f1(y, p), 2)})
    t = pd.DataFrame(rows).sort_values(["Backbone", "Protocol", "Repetition", "Fold"])
    os.makedirs(OUT_DIR, exist_ok=True)
    t.to_csv(os.path.join(OUT_DIR, "table_fold_level_accuracy.csv"), index=False)

    s = (t.groupby(["Backbone", "Protocol"])["Accuracy (%)"]
         .agg(Folds="count", Mean="mean", SD="std", Min="min", Max="max").round(2).reset_index())
    s.to_csv(os.path.join(OUT_DIR, "table_fold_level_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    print(f"\nSaved {len(t)} fold-level rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
