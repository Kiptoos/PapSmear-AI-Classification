"""
sli_bootstrap.py — Source-image Leakage Inflation (SLI) analysis for SIPaKMeD.

What it does
------------
1. Paired cluster bootstrap of SLI (image-level minus source-disjoint) for accuracy,
   balanced accuracy and macro-F1, resampling SOURCE IMAGES (all their cells together).
   The same bootstrap draw is applied to both protocols, so the CI is paired.
2. Per-class SLI on recall (sensitivity), with 95% CIs.
3. Calibration: ECE (15 bins) and Brier score per protocol, plus a paired
   cluster-bootstrap CI for the Brier difference.
4. Ranking stability: Kendall's tau between architecture rankings under the two
   protocols, and the bootstrap frequency with which the top-ranked model changes.
5. Pairwise architecture comparison within the source-disjoint protocol using the
   Nadeau-Bengio corrected resampled t-test (Holm-adjusted).

Input: one CSV of out-of-fold (OOF) predictions with columns
    model, protocol, fold, image_id, source_id, y_true, y_pred
optional:
    repeat            (CV repetition / seed index; default 0)
    prob_0..prob_{K-1} (softmax probabilities, needed for calibration)
protocol values must be exactly: image_level, source_disjoint
Every (model, repeat) must contain each image exactly once per protocol.

Usage
-----
python sli_bootstrap.py --oof oof_predictions.csv --out results/ --n_boot 5000
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROTOCOLS = ("image_level", "source_disjoint")


# --------------------------------------------------------------------------- #
# Metrics computed from confusion matrices (vectorised over a leading axis)
# --------------------------------------------------------------------------- #
def metrics_from_cm(cm):
    """cm: (..., K, K) with rows = true class. Returns dict of arrays."""
    cm = cm.astype(float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(-1)
    predicted = cm.sum(-2)
    total = cm.sum((-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(precision + np.nan_to_num(recall) > 0,
                      2 * precision * np.nan_to_num(recall)
                      / (precision + np.nan_to_num(recall)), 0.0)
    return {
        "accuracy": tp.sum(-1) / total,
        "balanced_accuracy": np.nanmean(recall, axis=-1),
        "macro_f1": f1.mean(-1),
        "recall": recall,  # (..., K)
    }


def ece_score(probs, y_true, n_bins=15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def ci(a, alpha=0.05):
    a = np.asarray(a)
    return np.nanpercentile(a, 100 * alpha / 2), np.nanpercentile(a, 100 * (1 - alpha / 2))


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def load_oof(path):
    df = pd.read_csv(path)
    if "repeat" not in df.columns:
        df["repeat"] = 0
    req = {"model", "protocol", "fold", "image_id", "source_id", "y_true", "y_pred"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"OOF file is missing columns: {missing}")
    bad = set(df["protocol"]) - set(PROTOCOLS)
    if bad:
        raise ValueError(f"Unknown protocol labels {bad}; use {PROTOCOLS}")

    # Each (model, repeat) must cover identical images under both protocols
    for (m, r), g in df.groupby(["model", "repeat"]):
        sets = [set(g.loc[g.protocol == p, "image_id"]) for p in PROTOCOLS]
        if sets[0] != sets[1]:
            raise ValueError(f"{m}, repeat {r}: image sets differ between protocols")
        if g.duplicated(["protocol", "image_id"]).any():
            raise ValueError(f"{m}, repeat {r}: duplicate image_id within a protocol")
    return df


def per_cluster_tables(df, K, clusters):
    """Per (model, protocol, repeat): per-cluster confusion counts (n_clusters, K*K)
    and per-cluster Brier sums (n_clusters,) if probabilities exist."""
    c_index = {c: i for i, c in enumerate(clusters)}
    prob_cols = [f"prob_{k}" for k in range(K)]
    has_probs = all(c in df.columns for c in prob_cols)
    tables = {}
    for key, g in df.groupby(["model", "protocol", "repeat"]):
        ci_ = g["source_id"].map(c_index).to_numpy()
        cell = g["y_true"].to_numpy() * K + g["y_pred"].to_numpy()
        cm = np.zeros((len(clusters), K * K))
        np.add.at(cm, (ci_, cell), 1)
        brier = None
        if has_probs:
            P = g[prob_cols].to_numpy()
            Y = np.eye(K)[g["y_true"].to_numpy()]
            per_img = ((P - Y) ** 2).sum(1)
            brier = np.zeros(len(clusters))
            np.add.at(brier, ci_, per_img)
        tables[key] = (cm, brier)
    return tables, has_probs


# --------------------------------------------------------------------------- #
# Main analyses
# --------------------------------------------------------------------------- #
def bootstrap_analysis(df, K, n_boot, seed, class_names):
    clusters = np.array(sorted(df["source_id"].unique()))
    n_c = len(clusters)
    tables, has_probs = per_cluster_tables(df, K, clusters)
    n_img_per_cluster = df.drop_duplicates("image_id").groupby("source_id").size() \
        .reindex(clusters).to_numpy()

    rng = np.random.default_rng(seed)
    # Bootstrap weights: how many times each source image is drawn
    W = rng.multinomial(n_c, np.full(n_c, 1 / n_c), size=n_boot).astype(float)
    W_point = np.ones((1, n_c))

    models = sorted(df["model"].unique())
    repeats = sorted(df["repeat"].unique())

    summary_rows, class_rows, calib_rows = [], [], []
    boot_store = {}  # (model, protocol, metric) -> bootstrap array for ranking

    for m in models:
        res = {}
        for p in PROTOCOLS:
            pt, bt, brier_pt, brier_bt = [], [], [], []
            for r in repeats:
                cm, brier = tables[(m, p, r)]
                pt.append(metrics_from_cm((W_point @ cm).reshape(1, K, K)))
                bt.append(metrics_from_cm((W @ cm).reshape(n_boot, K, K)))
                if has_probs:
                    n_pt = W_point @ n_img_per_cluster
                    brier_pt.append((W_point @ brier) / n_pt)
                    brier_bt.append((W @ brier) / (W @ n_img_per_cluster))
            # Average across CV repeats
            res[p] = {
                "point": {k: np.mean([d[k] for d in pt], axis=0) for k in pt[0]},
                "boot": {k: np.mean([d[k] for d in bt], axis=0) for k in bt[0]},
            }
            if has_probs:
                res[p]["brier_point"] = np.mean(brier_pt, axis=0)[0]
                res[p]["brier_boot"] = np.mean(brier_bt, axis=0)
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                boot_store[(m, p, metric)] = res[p]["boot"][metric]

        a, b = res["image_level"], res["source_disjoint"]
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            diff_boot = 100 * (a["boot"][metric] - b["boot"][metric])
            lo, hi = ci(diff_boot)
            lo_a, hi_a = ci(100 * a["boot"][metric])
            lo_b, hi_b = ci(100 * b["boot"][metric])
            summary_rows.append({
                "model": m, "metric": metric,
                "image_level_%": 100 * a["point"][metric][0],
                "image_level_CI": f"[{lo_a:.2f}, {hi_a:.2f}]",
                "source_disjoint_%": 100 * b["point"][metric][0],
                "source_disjoint_CI": f"[{lo_b:.2f}, {hi_b:.2f}]",
                "SLI_pp": 100 * (a["point"][metric][0] - b["point"][metric][0]),
                "SLI_CI_low": lo, "SLI_CI_high": hi,
                "P(SLI<=0)": np.mean(diff_boot <= 0),
            })

        # Per-class SLI on recall
        for k in range(K):
            diff_boot = 100 * (a["boot"]["recall"][:, k] - b["boot"]["recall"][:, k])
            lo, hi = ci(diff_boot)
            class_rows.append({
                "model": m, "class": class_names[k],
                "recall_image_level_%": 100 * a["point"]["recall"][0, k],
                "recall_source_disjoint_%": 100 * b["point"]["recall"][0, k],
                "SLI_recall_pp": 100 * (a["point"]["recall"][0, k] - b["point"]["recall"][0, k]),
                "CI_low": lo, "CI_high": hi,
            })

        # Calibration
        if has_probs:
            prob_cols = [f"prob_{k}" for k in range(K)]
            row = {"model": m}
            for p in PROTOCOLS:
                g = df[(df.model == m) & (df.protocol == p)]
                row[f"ECE_{p}"] = np.mean([
                    ece_score(gr[prob_cols].to_numpy(), gr["y_true"].to_numpy())
                    for _, gr in g.groupby("repeat")])
                row[f"Brier_{p}"] = res[p]["brier_point"]
            d = res["image_level"]["brier_boot"] - res["source_disjoint"]["brier_boot"]
            row["Brier_diff"] = row["Brier_image_level"] - row["Brier_source_disjoint"]
            row["Brier_diff_CI_low"], row["Brier_diff_CI_high"] = ci(d)
            calib_rows.append(row)

    # Ranking stability
    rank_rows = []
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        sub = pd.DataFrame(summary_rows)
        sub = sub[sub.metric == metric].set_index("model")
        tau, p_tau = stats.kendalltau(sub["image_level_%"], sub["source_disjoint_%"])
        A = np.stack([boot_store[(m, "image_level", metric)] for m in models], 1)
        B = np.stack([boot_store[(m, "source_disjoint", metric)] for m in models], 1)
        top_change = np.mean(A.argmax(1) != B.argmax(1))
        rank_rows.append({
            "metric": metric,
            "ranking_image_level": " > ".join(sub["image_level_%"].sort_values(ascending=False).index),
            "ranking_source_disjoint": " > ".join(sub["source_disjoint_%"].sort_values(ascending=False).index),
            "kendall_tau": tau, "tau_p": p_tau,
            "bootstrap_P(top_model_differs)": top_change,
        })

    return (pd.DataFrame(summary_rows), pd.DataFrame(class_rows),
            pd.DataFrame(calib_rows), pd.DataFrame(rank_rows))


def nadeau_bengio(df, K, protocol="source_disjoint", metric="balanced_accuracy"):
    """Corrected resampled t-test for pairwise architecture comparison on the SAME folds."""
    sub = df[df.protocol == protocol]
    n_folds = sub["fold"].nunique()
    rows = []
    fold_scores = {}
    for (m, r, f), g in sub.groupby(["model", "repeat", "fold"]):
        cm = np.zeros((K, K))
        np.add.at(cm, (g.y_true, g.y_pred), 1)
        fold_scores[(m, r, f)] = metrics_from_cm(cm[None])[metric][0]
    models = sorted(sub["model"].unique())
    for m1, m2 in itertools.combinations(models, 2):
        keys = sorted({(r, f) for (mm, r, f) in fold_scores if mm == m1})
        d = np.array([fold_scores[(m1, r, f)] - fold_scores[(m2, r, f)] for r, f in keys])
        J = len(d)
        ratio = 1 / (n_folds - 1)  # n_test / n_train for k-fold CV
        var = d.var(ddof=1) * (1 / J + ratio)
        t = d.mean() / np.sqrt(var) if var > 0 else np.inf * np.sign(d.mean())
        p = 2 * stats.t.sf(abs(t), df=J - 1)
        rows.append({"model_A": m1, "model_B": m2, "metric": metric,
                     "mean_diff_pp": 100 * d.mean(), "t": t, "df": J - 1, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        order = np.argsort(out.p.values)
        m = len(out)
        adj = np.empty(m)
        running = 0
        for rank, i in enumerate(order):
            running = max(running, min(1, (m - rank) * out.p.values[i]))
            adj[i] = running
        out["p_holm"] = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--out", default="results_sli")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_names", default="Dyskeratotic,Koilocytotic,Metaplastic,"
                    "Parabasal,Superficial-Intermediate",
                    help="Comma-separated, in label-index order used in training")
    args = ap.parse_args()

    class_names = args.class_names.split(",")
    K = len(class_names)
    df = load_oof(args.oof)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{df.model.nunique()} models | {df.source_id.nunique()} source images | "
          f"{df.image_id.nunique()} cells | {df.repeat.nunique()} repeat(s) | "
          f"{args.n_boot} bootstrap draws")

    summary, per_class, calib, ranking = bootstrap_analysis(
        df, K, args.n_boot, args.seed, class_names)
    nb = nadeau_bengio(df, K)

    summary.round(3).to_csv(out / "sli_summary.csv", index=False)
    per_class.round(3).to_csv(out / "sli_per_class.csv", index=False)
    ranking.round(4).to_csv(out / "ranking_stability.csv", index=False)
    nb.round(4).to_csv(out / "pairwise_nadeau_bengio_source_disjoint.csv", index=False)
    if len(calib):
        calib.round(4).to_csv(out / "calibration.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== SLI (percentage points), paired source-image cluster bootstrap ===")
    print(summary[summary.metric == "accuracy"]
          [["model", "image_level_%", "source_disjoint_%", "SLI_pp",
            "SLI_CI_low", "SLI_CI_high", "P(SLI<=0)"]].round(2).to_string(index=False))
    print("\n=== Ranking stability ===")
    print(ranking.round(3).to_string(index=False))
    print(f"\nAll tables written to {out.resolve()}")


if __name__ == "__main__":
    main()
