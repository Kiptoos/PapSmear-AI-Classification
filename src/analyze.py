"""
Step 3 - every table, figure and statistic for the revised manuscript,
computed from the pooled out-of-fold predictions only.

    python src\\analyze.py

Writes tables to outputs/results/, figures to outputs/figures/, and a
plain-text digest outputs/results/analysis_summary.txt.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, roc_curve)

from common import (CLASS_NAMES, CATEGORIES, CATEGORY_OF_CLASS, STRATEGIES,
                    PRED_DIR, RESULTS_DIR, FIG_DIR)

PROB_COLS = [f"prob_{c}" for c in CLASS_NAMES]
N_BOOT = 5000
RNG = np.random.default_rng(2026)
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ABN_IDX = [CLASS_NAMES.index(c) for c in CLASS_NAMES if CATEGORY_OF_CLASS[c] == "Abnormal"]
CAT_MEMBERS = [[CLASS_NAMES.index(c) for c in CLASS_NAMES if CATEGORY_OF_CLASS[c] == cat]
               for cat in CATEGORIES]
out_lines = []


def say(s=""):
    print(s)
    out_lines.append(s)


def pm(vals, scale=100, nd=2):
    vals = np.asarray(vals, dtype=float)
    return f"{vals.mean() * scale:.{nd}f} ± {vals.std(ddof=1 if len(vals) > 1 else 0) * scale:.{nd}f}"


# ---------------------------------------------------------------- metrics
def ece(prob, y, n_bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def brier(prob, y):
    onehot = np.eye(prob.shape[1])[y]
    return float(((prob - onehot) ** 2).sum(1).mean())


def risk_coverage(prob, y):
    conf = prob.max(1)
    order = np.argsort(-conf, kind="stable")
    correct = (prob.argmax(1) == y)[order]
    n = len(y)
    cov = np.arange(1, n + 1) / n
    acc = np.cumsum(correct) / np.arange(1, n + 1)
    aurc = float(np.mean(1 - acc))
    return cov, acc, aurc


def acc_at_coverage(cov, acc, c):
    return float(acc[max(0, int(np.ceil(c * len(cov))) - 1)])


def collapse(prob, members):
    return np.stack([prob[:, m].sum(1) for m in members], axis=1)


def basic_metrics(prob, y):
    pred = prob.argmax(1)
    cov, acc_curve, aurc = risk_coverage(prob, y)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "macro_auc_ovr": roc_auc_score(y, prob, multi_class="ovr", average="macro"),
        "ece": ece(prob, y),
        "brier": brier(prob, y),
        "aurc": aurc,
    }


# ---------------------------------------------------------- bootstrap
def group_bootstrap_diff(diff_per_image, groups):
    """CI and two-sided p for mean(diff), resampling source images (clusters)."""
    g_codes, g_idx = np.unique(groups, return_inverse=True)
    g_sum = np.bincount(g_idx, weights=diff_per_image)
    g_cnt = np.bincount(g_idx).astype(float)
    G = len(g_codes)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = RNG.integers(0, G, G)
        boots[b] = g_sum[s].sum() / g_cnt[s].sum()
    est = diff_per_image.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return est, lo, hi, max(p, 1 / N_BOOT)


def holm(pvals):
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------- main
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*_oof.csv")))
    if not files:
        raise SystemExit("No prediction files found - run run_cv.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    archs = list(dict.fromkeys(df["arch"]))

    # integrity check: one prediction per image per (arch, strategy, repeat)
    counts = df.groupby(["arch", "strategy", "repeat"])["filename"].agg(["count", "nunique"])
    assert (counts["count"] == counts["nunique"]).all()
    n_images = int(counts["nunique"].iloc[0])
    groups_all = df.drop_duplicates("filename").set_index("filename")["group"]

    say("=" * 72)
    say("PapSmear_AI v3 - analysis summary (pooled out-of-fold predictions)")
    say("=" * 72)
    say(f"Images: {n_images}   Source images (groups): {groups_all.nunique()}")
    gsize = groups_all.value_counts()
    say(f"Crops per source image: mean {gsize.mean():.2f}, median {gsize.median():.0f}, "
        f"max {gsize.max()}; source images with >=2 crops: {(gsize >= 2).sum()} "
        f"({(gsize >= 2).mean() * 100:.1f}%)")
    say(f"Architectures: {', '.join(archs)}")
    say(f"Repeats per strategy: {df['repeat'].nunique()}  (5 folds each)")

    # ---------------- fold composition + leakage exposure (R2 ED1)
    ref = df[df["arch"] == archs[0]]
    comp_rows = []
    for (strategy, rep), sub in ref.groupby(["strategy", "repeat"]):
        for fold, fsub in sub.groupby("fold"):
            train_groups = set(sub.loc[sub["fold"] != fold, "group"])
            exposed = fsub["group"].isin(train_groups)
            row = {"strategy": strategy, "repeat": rep, "fold": fold + 1,
                   "n_val_images": len(fsub), "n_val_groups": fsub["group"].nunique(),
                   "val_images_sharing_source_with_train": int(exposed.sum()),
                   "pct_val_images_exposed": round(exposed.mean() * 100, 1)}
            for k, c in enumerate(CLASS_NAMES):
                row[f"n_{c}"] = int((fsub["true_label"] == k).sum())
            comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(RESULTS_DIR, "fold_composition.csv"), index=False)
    say("\n[Fold composition] (full table: fold_composition.csv)")
    for s in STRATEGIES:
        cs = comp[comp["strategy"] == s]
        say(f"  {s:15s}: val images/fold {cs.n_val_images.min()}-{cs.n_val_images.max()}, "
            f"val source images/fold {cs.n_val_groups.min()}-{cs.n_val_groups.max()}, "
            f"val images whose source image is also in training: "
            f"{cs.pct_val_images_exposed.mean():.1f}% on average")

    # ---------------- main table: per arch x strategy, mean ± SD over repeats
    per_rep = []
    for (arch, strategy, rep), sub in df.groupby(["arch", "strategy", "repeat"]):
        sub = sub.sort_values("filename")
        m = basic_metrics(sub[PROB_COLS].values, sub["true_label"].values)
        m.update(arch=arch, strategy=strategy, repeat=rep)
        per_rep.append(m)
    per_rep = pd.DataFrame(per_rep)
    per_rep.to_csv(os.path.join(RESULTS_DIR, "metrics_per_repeat.csv"), index=False)

    say("\n[Table: accuracy, macro-F1, macro AUC (%) - mean ± SD over repeats]")
    main_rows = []
    for arch in archs:
        for s in STRATEGIES:
            pr = per_rep[(per_rep.arch == arch) & (per_rep.strategy == s)]
            main_rows.append({"Architecture": arch, "Split": s,
                              "Accuracy (%)": pm(pr.accuracy), "Macro-F1 (%)": pm(pr.macro_f1),
                              "Macro AUC (OvR)": pm(pr.macro_auc_ovr, 1, 4),
                              "ECE": pm(pr.ece, 1, 4), "Brier": pm(pr.brier, 1, 4),
                              "AURC": pm(pr.aurc, 1, 4)})
    main_tab = pd.DataFrame(main_rows)
    main_tab.to_csv(os.path.join(RESULTS_DIR, "table_main_metrics.csv"), index=False)
    say(main_tab.to_string(index=False))

    # ---------------- SLI with group bootstrap (headline result)
    say("\n[Source-image Leakage Inflation (SLI) = naive - source-disjoint accuracy, pp]")
    say("  per-image correctness averaged over repeats; 95% CI and p from bootstrap over source images")
    correct_mean = {}
    for (arch, s), sub in df.groupby(["arch", "strategy"]):
        sub = sub.assign(correct=(sub[PROB_COLS].values.argmax(1) == sub["true_label"]).astype(float))
        correct_mean[(arch, s)] = sub.groupby("filename")["correct"].mean().sort_index()
    order_fn = correct_mean[(archs[0], "naive")].index
    g_vec = groups_all.loc[order_fn].values
    sli_rows = []
    for arch in archs:
        diff = (correct_mean[(arch, "naive")].loc[order_fn] -
                correct_mean[(arch, "source_disjoint")].loc[order_fn]).values
        est, lo, hi, p = group_bootstrap_diff(diff, g_vec)
        rep_sli = (per_rep[(per_rep.arch == arch) & (per_rep.strategy == "naive")].sort_values("repeat").accuracy.values -
                   per_rep[(per_rep.arch == arch) & (per_rep.strategy == "source_disjoint")].sort_values("repeat").accuracy.values)
        sli_rows.append({"Architecture": arch, "SLI (pp)": round(est * 100, 2),
                         "95% CI low": round(lo * 100, 2), "95% CI high": round(hi * 100, 2),
                         "p (bootstrap)": round(p, 4),
                         "SLI per-repeat mean ± SD (pp)": pm(rep_sli)})
    sli = pd.DataFrame(sli_rows)
    sli.to_csv(os.path.join(RESULTS_DIR, "table_sli.csv"), index=False)
    say(sli.to_string(index=False))

    # ---------------- pairwise architecture comparison (source-disjoint)
    say("\n[Pairwise accuracy differences, source-disjoint split (pp), Holm-adjusted]")
    pair_rows = []
    for i, a in enumerate(archs):
        for b in archs[i + 1:]:
            diff = (correct_mean[(a, "source_disjoint")].loc[order_fn] -
                    correct_mean[(b, "source_disjoint")].loc[order_fn]).values
            est, lo, hi, p = group_bootstrap_diff(diff, g_vec)
            pair_rows.append({"A": a, "B": b, "A - B (pp)": round(est * 100, 2),
                              "95% CI low": round(lo * 100, 2), "95% CI high": round(hi * 100, 2),
                              "p": p})
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        pairs["p_holm"] = holm(pairs["p"].values).round(4)
        pairs["p"] = pairs["p"].round(4)
        pairs.to_csv(os.path.join(RESULTS_DIR, "table_pairwise.csv"), index=False)
        say(pairs.to_string(index=False))

    # ---------------- primary model = best mean source-disjoint accuracy
    sd = per_rep[per_rep.strategy == "source_disjoint"].groupby("arch").accuracy.mean()
    primary = sd.idxmax()
    say(f"\nPrimary model (highest source-disjoint accuracy): {primary}")

    # per-class metrics for all archs (source-disjoint), mean over repeats
    pc_rows = []
    for arch in archs:
        reps = []
        for rep, sub in df[(df.arch == arch) & (df.strategy == "source_disjoint")].groupby("repeat"):
            p_, r_, f_, s_ = precision_recall_fscore_support(
                sub.true_label, sub[PROB_COLS].values.argmax(1), labels=range(len(CLASS_NAMES)), zero_division=0)
            reps.append((p_, r_, f_, s_))
        reps = np.array(reps, dtype=float)
        for k, c in enumerate(CLASS_NAMES):
            pc_rows.append({"Architecture": arch, "Class": c,
                            "Precision": pm(reps[:, 0, k], 1, 3), "Recall": pm(reps[:, 1, k], 1, 3),
                            "F1": pm(reps[:, 2, k], 1, 3), "Support": int(reps[0, 3, k])})
    pc = pd.DataFrame(pc_rows)
    pc.to_csv(os.path.join(RESULTS_DIR, "table_per_class_source_disjoint.csv"), index=False)
    say(f"\n[Per-class metrics, {primary}, source-disjoint (mean ± SD over repeats)]")
    say(pc[pc.Architecture == primary].drop(columns="Architecture").to_string(index=False))

    # ---------------- primary model: granularity, ROC, calibration, selective
    prim = df[(df.arch == primary) & (df.strategy == "source_disjoint")]
    gran_rows, sel_rows = [], []
    for rep, sub in prim.groupby("repeat"):
        y5 = sub.true_label.values
        P5 = sub[PROB_COLS].values
        y3 = np.array([CATEGORIES.index(CATEGORY_OF_CLASS[CLASS_NAMES[k]]) for k in y5])
        P3 = collapse(P5, CAT_MEMBERS)
        yb = np.isin(y5, ABN_IDX).astype(int)
        sb = P5[:, ABN_IDX].sum(1)
        pb = (sb >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
        gran_rows.append({
            "repeat": rep,
            "5class_acc": accuracy_score(y5, P5.argmax(1)),
            "5class_macroF1": f1_score(y5, P5.argmax(1), average="macro"),
            "3cat_acc": accuracy_score(y3, P3.argmax(1)),
            "3cat_macroF1": f1_score(y3, P3.argmax(1), average="macro"),
            "binary_acc": accuracy_score(yb, pb),
            "binary_sensitivity": tp / (tp + fn), "binary_specificity": tn / (tn + fp),
            "binary_auc": roc_auc_score(yb, sb),
        })
        cov, acc_c, aurc = risk_coverage(P5, y5)
        sel = {"repeat": rep, "AURC": aurc}
        for c in COVERAGES:
            sel[f"acc@{int(c * 100)}%cov"] = acc_at_coverage(cov, acc_c, c)
        sel_rows.append(sel)
    gran = pd.DataFrame(gran_rows)
    gran.to_csv(os.path.join(RESULTS_DIR, "granularity_per_repeat.csv"), index=False)
    say(f"\n[Granularity - {primary}, source-disjoint, mean ± SD over repeats (%)]")
    say("  Categories: Normal = Superficial-Intermediate + Parabasal; Benign = Metaplastic; "
        "Abnormal = Koilocytotic + Dyskeratotic")
    say("  Binary: Abnormal vs non-abnormal (Normal + Benign); score = P(Dysk) + P(Koilo), threshold 0.5")
    for col in gran.columns[1:]:
        scale = 1 if col.endswith("auc") else 100
        say(f"  {col:22s}: {pm(gran[col], scale, 4 if scale == 1 else 2)}")
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(os.path.join(RESULTS_DIR, "selective_classification_per_repeat.csv"), index=False)
    say(f"\n[Selective classification - {primary}, source-disjoint (accuracy %, at fixed coverage)]")
    say("  No threshold is selected on evaluation data; these are points on the risk-coverage curve.")
    for col in sel_df.columns[1:]:
        say(f"  {col:16s}: {pm(sel_df[col], 1 if col == 'AURC' else 100, 4 if col == 'AURC' else 2)}")

    # ----------------------------- figures (repeat 0 of the primary model)
    r0 = prim[prim.repeat == 0]
    y5, P5 = r0.true_label.values, r0[PROB_COLS].values

    # Confusion matrix
    cm = confusion_matrix(y5, P5.argmax(1), labels=range(len(CLASS_NAMES)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(RESULTS_DIR, f"confusion_matrix_{primary}_repeat1.csv"))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(5), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"{primary}, source-disjoint CV (n = {cm.sum()})")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_confusion_matrix.png"), dpi=300); plt.close(fig)

    # ROC curves: one-vs-rest five-class + binary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve((y5 == k).astype(int), P5[:, k])
        axes[0].plot(fpr, tpr, label=f"{c} (AUC = {roc_auc_score((y5 == k).astype(int), P5[:, k]):.3f})")
    yb = np.isin(y5, ABN_IDX).astype(int); sb = P5[:, ABN_IDX].sum(1)
    fpr, tpr, _ = roc_curve(yb, sb)
    axes[1].plot(fpr, tpr, label=f"Abnormal vs non-abnormal (AUC = {roc_auc_score(yb, sb):.3f})")
    for ax, t in zip(axes, ["Five-class, one-vs-rest", "Binary"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(t); ax.legend(loc="lower right", fontsize=8); ax.set_aspect("equal")
    fig.suptitle(f"{primary}, out-of-fold probabilities, source-disjoint CV (n = {len(y5)})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_roc.png"), dpi=300); plt.close(fig)

    # Reliability diagram
    conf, correct = P5.max(1), (P5.argmax(1) == y5)
    edges = np.linspace(0, 1, 16)
    mids, accs, cnts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            mids.append(conf[m].mean()); accs.append(correct[m].mean()); cnts.append(m.sum())
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mids, accs, "o-", label=f"{primary} (ECE = {ece(P5, y5):.3f})")
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability diagram (15 bins)"); ax.legend(); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_reliability.png"), dpi=300); plt.close(fig)

    # Risk-coverage
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for arch in archs:
        sub = df[(df.arch == arch) & (df.strategy == "source_disjoint") & (df.repeat == 0)]
        cov, acc_c, aurc = risk_coverage(sub[PROB_COLS].values, sub.true_label.values)
        ax.plot(cov * 100, acc_c * 100, label=f"{arch} (AURC = {aurc:.4f})")
    ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Accuracy on retained samples (%)")
    ax.set_title("Selective classification (source-disjoint CV)")
    ax.invert_xaxis(); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_risk_coverage.png"), dpi=300); plt.close(fig)

    # Naive vs source-disjoint with SLI CIs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(archs)); w = 0.38
    for off, s, lab in [(-w / 2, "naive", "Naive (image-level)"), (w / 2, "source_disjoint", "Source-disjoint")]:
        pr = [per_rep[(per_rep.arch == a) & (per_rep.strategy == s)].accuracy for a in archs]
        axes[0].bar(x + off, [v.mean() * 100 for v in pr], w,
                    yerr=[v.std(ddof=1) * 100 if len(v) > 1 else 0 for v in pr], capsize=3, label=lab)
    axes[0].set_xticks(x, archs, rotation=20); axes[0].set_ylabel("Pooled OOF accuracy (%)")
    lo_all = per_rep.accuracy.min() * 100
    axes[0].set_ylim(np.floor(lo_all) - 2, 100); axes[0].legend(); axes[0].set_title("Accuracy by split strategy")
    axes[1].errorbar(x, sli["SLI (pp)"], yerr=[sli["SLI (pp)"] - sli["95% CI low"],
                                               sli["95% CI high"] - sli["SLI (pp)"]],
                     fmt="o", capsize=4)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xticks(x, archs, rotation=20); axes[1].set_ylabel("SLI (percentage points)")
    axes[1].set_title("Source-image leakage inflation, 95% bootstrap CI")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_sli.png"), dpi=300); plt.close(fig)

    # release file: one OOF prediction per image (repeat 1, every arch & strategy)
    release = df[df.repeat == 0].drop(columns=["seed"])
    release.to_csv(os.path.join(RESULTS_DIR, "oof_predictions_repeat1_all_models.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "analysis_summary.json"), "w") as f:
        json.dump({"primary_model": primary, "n_images": n_images,
                   "n_groups": int(groups_all.nunique()), "archs": archs}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "analysis_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\nDone. Send me outputs/results/analysis_summary.txt and the outputs/figures/ PNGs.")


if __name__ == "__main__":
    main()
