"""
Step 3a - combine the per-architecture OOF files from run_cv.py into the single
file that sli_bootstrap.py expects.

    python src\\prepare_oof.py
    python src\\sli_bootstrap.py --oof outputs\\results\\oof_all_for_bootstrap.csv --out outputs\\results\\sli_bootstrap

Mapping: arch -> model, strategy naive -> image_level,
filename -> image_id, group -> source_id, true_label -> y_true,
argmax of the probabilities -> y_pred, prob_<class> -> prob_0..prob_4.
"""
import glob
import os

import pandas as pd

from common import CLASS_NAMES, PRED_DIR, RESULTS_DIR

OUT = os.path.join(RESULTS_DIR, "oof_all_for_bootstrap.csv")


def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*_oof.csv")))
    if not files:
        raise SystemExit(f"No *_oof.csv files in {PRED_DIR}")
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    parts = []
    for f in files:
        d = pd.read_csv(f)
        out = pd.DataFrame({
            "model": d["arch"],
            "protocol": d["strategy"].replace({"naive": "image_level"}),
            "repeat": d["repeat"],
            "fold": d["fold"],
            "image_id": d["filename"],
            "source_id": d["group"],
            "y_true": d["true_label"].astype(int),
            "y_pred": d[prob_cols].to_numpy().argmax(1),
        })
        for k, c in enumerate(prob_cols):
            out[f"prob_{k}"] = d[c]
        parts.append(out)
        print(f"{os.path.basename(f)}: {len(d)} rows, "
              f"{d['repeat'].nunique()} repeats, strategies {sorted(d['strategy'].unique())}")
    allp = pd.concat(parts, ignore_index=True)
    allp.to_csv(OUT, index=False)
    print(f"\nSaved {len(allp)} rows -> {OUT}")


if __name__ == "__main__":
    main()
