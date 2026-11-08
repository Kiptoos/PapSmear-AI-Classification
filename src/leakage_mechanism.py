"""
Step 6 - mechanism of source-image leakage and the case for cluster-level inference.
Uses the existing out-of-fold predictions only (no retraining).

    python src\\leakage_mechanism.py

A. Dose-response (does inflation grow with the number of sibling cells in training?)
   For every image and backbone, the paired difference
       d_i = mean over repeats of [correct under naive] - [correct under source-disjoint]
   is the image's own contribution to the SLI (same image, same backbone, so image
   difficulty cancels out). Under source-disjoint splitting no image has siblings in
   training; under naive splitting an image has (group size - group members in its
   validation fold) siblings in training.
   * Negative control: cells from single-cell source images have no siblings under
     either protocol, so their SLI should be about zero.
   * SLI by source-image size bin, with source-image cluster bootstrap CIs.
   * Slope of d_i on the number of training siblings, adjusted for class
     (class fixed effects), with a cluster bootstrap CI.

B. Cluster vs image-level bootstrap
   The SLI and source-disjoint accuracy are bootstrapped twice: resampling whole
   source images (cells not independent) and resampling cells as if independent.
   Width ratio > 1 and design effect = ratio^2 show how much an image-level
   analysis would understate uncertainty.

Outputs: outputs/results/leakage_mechanism/*.csv
"""
import glob
import os

import numpy as np
import pandas as pd

PRED_DIR = os.path.join("outputs", "predictions")
OUT_DIR = os.path.join("outputs", "results", "leakage_mechanism")
CLASS_NAMES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
PROB_COLS = [f"prob_{c}" for c in CLASS_NAMES]
N_BOOT = 2000
SEED = 42
BINS = [(1, 1, "1 (single-cell source)"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 10, "7-10"), (11, 999, "11+")]


def per_image_table(df):
    """One row per image: paired difference d, training siblings under naive, class, group."""
    df = df.copy()
    df["correct"] = (df[PROB_COLS].to_numpy().argmax(1) == df["true_label"].to_numpy()).astype(float)
    gsize = df[(df.strategy == "naive") & (df.repeat == df.repeat.min())].groupby("group").size()
    df["group_size"] = df["group"].map(gsize)
    nv = df[df.strategy == "naive"].copy()
    in_val = nv.groupby(["repeat", "fold", "group"])["filename"].transform("size")
    nv["train_siblings"] = nv["group_size"] - in_val
    a = nv.groupby("filename").agg(correct_naive=("correct", "mean"),
                                   train_siblings=("train_siblings", "mean"),
                                   group=("group", "first"), true_label=("true_label", "first"),
                                   group_size=("group_size", "first"))
    sd = df[df.strategy == "source_disjoint"].groupby("filename")["correct"].mean().rename("correct_sd")
    t = a.join(sd)
    t["d"] = t["correct_naive"] - t["correct_sd"]
    return t.reset_index()


def cluster_weights(groups, n_boot, rng):
    uniq, inv = np.unique(groups, return_inverse=True)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot).astype(float)
    return W[:, inv]  # (n_boot, n_images): weight of each image in each draw


def image_weights(n, n_boot, rng):
    return rng.multinomial(n, np.full(n, 1 / n), size=n_boot).astype(float)


def wmean(W, v):
    return (W @ v) / W.sum(1)


def ci(a):
    return np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)


def slope_with_class(W, x, d, cls):
    """Weighted OLS of d on x with class fixed effects; returns slope per draw."""
    X = np.column_stack([x] + [(cls == k).astype(float) for k in range(5)])
    out = np.empty(W.shape[0])
    for b in range(W.shape[0]):
        w = W[b]
        XtW = X.T * w
        try:
            out[b] = np.linalg.solve(XtW @ X, XtW @ d)[0]
        except np.linalg.LinAlgError:
            out[b] = np.nan
    return out


def analyse(arch, t, rng):
    n = len(t)
    d = t["d"].to_numpy()
    x = t["train_siblings"].to_numpy()
    cls = t["true_label"].to_numpy()
    groups = t["group"].to_numpy()
    Wc = cluster_weights(groups, N_BOOT, rng)
    Wi = image_weights(n, N_BOOT, rng)

    # ---- A1/A2: SLI by source-image size bin
    bins = []
    for lo, hi, lab in BINS:
        m = (t.group_size >= lo) & (t.group_size <= hi)
        if m.sum() == 0:
            continue
        sub = Wc[:, m.to_numpy()]
        boot = 100 * (sub @ d[m.to_numpy()]) / sub.sum(1)
        lo_ci, hi_ci = ci(boot)
        bins.append({"Backbone": arch, "Source-image size": lab, "Cells": int(m.sum()),
                     "Mean training siblings (naive)": round(float(x[m.to_numpy()].mean()), 2),
                     "Naive acc (%)": round(100 * t.loc[m, "correct_naive"].mean(), 2),
                     "Source-disjoint acc (%)": round(100 * t.loc[m, "correct_sd"].mean(), 2),
                     "SLI (pp)": round(100 * d[m.to_numpy()].mean(), 2),
                     "CI low": round(lo_ci, 2), "CI high": round(hi_ci, 2)})

    # ---- A3: dose-response slope, class-adjusted
    full = np.ones((1, n))
    slope = 100 * slope_with_class(full, x, d, cls)[0]
    sb = 100 * slope_with_class(Wc, x, d, cls)
    s_lo, s_hi = ci(sb)
    dose = {"Backbone": arch, "Slope (pp per training sibling, class-adjusted)": round(slope, 3),
            "CI low": round(s_lo, 3), "CI high": round(s_hi, 3), "P(slope<=0)": round(float(np.mean(sb <= 0)), 4)}

    # ---- B: cluster vs image bootstrap
    sli_c, sli_i = 100 * wmean(Wc, d), 100 * wmean(Wi, d)
    acc = t["correct_sd"].to_numpy()
    acc_c, acc_i = 100 * wmean(Wc, acc), 100 * wmean(Wi, acc)
    rows = []
    for name, pt, bc, bi in [("SLI (pp)", 100 * d.mean(), sli_c, sli_i),
                             ("Source-disjoint accuracy (%)", 100 * acc.mean(), acc_c, acc_i)]:
        c_lo, c_hi = ci(bc)
        i_lo, i_hi = ci(bi)
        ratio = (c_hi - c_lo) / (i_hi - i_lo)
        rows.append({"Backbone": arch, "Quantity": name, "Estimate": round(pt, 2),
                     "Cluster CI": f"{c_lo:.2f} to {c_hi:.2f}", "Image-level CI": f"{i_lo:.2f} to {i_hi:.2f}",
                     "Width ratio (cluster/image)": round(ratio, 2), "Design effect": round(ratio ** 2, 2)})
    return bins, dose, rows


def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*_oof.csv")))
    if not files:
        raise SystemExit(f"No *_oof.csv in {PRED_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_bins, all_dose, all_boot, pooled = [], [], [], []
    for f in files:
        df = pd.read_csv(f)
        arch = df["arch"].iloc[0]
        t = per_image_table(df)
        pooled.append(t.assign(arch=arch))
        b, d, r = analyse(arch, t, rng)
        all_bins += b; all_dose.append(d); all_boot += r
        print(f"done: {arch}")

    # Pooled over backbones: average each image's d across the six backbones
    p = pd.concat(pooled)
    tp = p.groupby("filename").agg(d=("d", "mean"), correct_naive=("correct_naive", "mean"),
                                   correct_sd=("correct_sd", "mean"),
                                   train_siblings=("train_siblings", "mean"), group=("group", "first"),
                                   true_label=("true_label", "first"), group_size=("group_size", "first")).reset_index()
    b, d, r = analyse("All backbones (mean)", tp, rng)
    all_bins += b; all_dose.append(d); all_boot += r

    bins, dose, boot = pd.DataFrame(all_bins), pd.DataFrame(all_dose), pd.DataFrame(all_boot)
    bins.to_csv(os.path.join(OUT_DIR, "sli_by_source_size.csv"), index=False)
    dose.to_csv(os.path.join(OUT_DIR, "dose_response_slope.csv"), index=False)
    boot.to_csv(os.path.join(OUT_DIR, "cluster_vs_image_bootstrap.csv"), index=False)

    pd.set_option("display.width", 250)
    print("\n=== A. SLI by source-image size (pooled over backbones) ===")
    print(bins[bins.Backbone == "All backbones (mean)"].drop(columns="Backbone").to_string(index=False))
    print("\n=== A. Single-cell source images (negative control), per backbone ===")
    print(bins[bins["Source-image size"].str.startswith("1")][["Backbone", "Cells", "SLI (pp)", "CI low", "CI high"]]
          .to_string(index=False))
    print("\n=== A. Dose-response slope (class-adjusted) ===")
    print(dose.to_string(index=False))
    print("\n=== B. Cluster vs image-level bootstrap ===")
    print(boot.to_string(index=False))
    print(f"\nTables saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
