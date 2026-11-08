"""
Step 5 - secondary analyses on the saved Herlev results (no retraining).

    python src\\herlev_secondary.py

Reads outputs/results/herlev/<arch>_per_image.csv and <arch>_summary.json and writes:
  table_secondary_no_columnar.csv   metrics excluding normal columnar cells
                                    (a cell type absent from SIPaKMeD), 95% CIs
  table_internal_vs_external.csv    leakage gap vs external drop per architecture
  table_rank_agreement.csv          Spearman / Kendall between internal and external
  table_class_mapping_all_archs.csv which SIPaKMeD class each Herlev class is mapped to
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

HERLEV_DIR = os.path.join("outputs", "results", "herlev")
N_BOOT = 1000
SEED = 42


def metrics(y, s):
    p = s >= 0.5
    sens = (p & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = (~p & (y == 0)).sum() / max((y == 0).sum(), 1)
    return np.array([sens, spec, (sens + spec) / 2, roc_auc_score(y, s)])


def seed_mean(y, S):
    return np.mean([metrics(y, S[:, k]) for k in range(S.shape[1])], axis=0)


def boot_ci(y, S, rng):
    n = len(y)
    draws = np.array([seed_mean(y[i], S[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))])
    return np.percentile(draws, [2.5, 97.5], axis=0)


def main():
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(HERLEV_DIR, "*_per_image.csv")))
    if not files:
        raise SystemExit(f"No per-image files in {HERLEV_DIR}")
    names = ["Sensitivity", "Specificity", "Balanced accuracy", "AUC"]
    sec_rows, ie_rows, maps = [], [], []
    for f in files:
        arch = os.path.basename(f).replace("_per_image.csv", "")
        d = pd.read_csv(f)
        seeds = [c for c in d.columns if c.startswith("p_abnormal_seed")]
        for label, sub in [("All cells (primary)", d),
                           ("Excluding columnar (secondary)", d[d.herlev_class != "Normal columnar"])]:
            y, S = sub.y_binary.to_numpy(), sub[seeds].to_numpy()
            pt, ci = seed_mean(y, S), boot_ci(y, S, rng)
            row = {"Architecture": arch, "Analysis": label, "n": len(sub)}
            for k, nm in enumerate(names):
                if nm == "AUC":
                    row[nm] = f"{pt[k]:.3f} [{ci[0, k]:.3f}, {ci[1, k]:.3f}]"
                else:
                    row[f"{nm} (%)"] = f"{100 * pt[k]:.1f} [{100 * ci[0, k]:.1f}, {100 * ci[1, k]:.1f}]"
            sec_rows.append(row)
            if label.startswith("Excluding"):
                ext_nc = pt

        s = json.load(open(os.path.join(HERLEV_DIR, f"{arch}_summary.json")))
        naive = s["internal"]["naive"]["point"]
        sd = s["internal"]["source_disjoint"]["point"]
        ext = s["external_herlev"]["point"]
        ie_rows.append({
            "Architecture": arch,
            "Internal naive BA (%)": 100 * naive["balanced_accuracy"],
            "Internal source-disjoint BA (%)": 100 * sd["balanced_accuracy"],
            "External Herlev BA (%)": 100 * ext["balanced_accuracy"],
            "Leakage gap (pp)": 100 * (naive["balanced_accuracy"] - sd["balanced_accuracy"]),
            "External drop (pp)": 100 * (sd["balanced_accuracy"] - ext["balanced_accuracy"]),
            "External drop excl. columnar (pp)": 100 * (sd["balanced_accuracy"] - ext_nc[2]),
            "Internal source-disjoint AUC": sd["auc"],
            "External Herlev AUC": ext["auc"],
            "External Herlev BA excl. columnar (%)": 100 * ext_nc[2],
            "External Herlev AUC excl. columnar": ext_nc[3],
        })

        ct = pd.crosstab(d.herlev_class, d.pred_5class_seed0, normalize="index") * 100
        ct.insert(0, "Architecture", arch)
        maps.append(ct.round(1).reset_index())

    out = lambda df, name: (df.to_csv(os.path.join(HERLEV_DIR, name), index=False), name)[1]
    sec = pd.DataFrame(sec_rows)
    ie = pd.DataFrame(ie_rows).round(3)
    rank_rows = []
    for a, b in [("Internal source-disjoint BA (%)", "External Herlev BA (%)"),
                 ("Internal source-disjoint AUC", "External Herlev AUC"),
                 ("Internal source-disjoint BA (%)", "External Herlev BA excl. columnar (%)"),
                 ("Internal source-disjoint AUC", "External Herlev AUC excl. columnar")]:
        rho, p_rho = spearmanr(ie[a], ie[b])
        tau, p_tau = kendalltau(ie[a], ie[b])
        rank_rows.append({"Internal": a, "External": b, "Spearman rho": round(rho, 3),
                          "p": round(p_rho, 3), "Kendall tau": round(tau, 3), "p_tau": round(p_tau, 3),
                          "n_architectures": len(ie)})
    rank = pd.DataFrame(rank_rows)
    mp = pd.concat(maps, ignore_index=True).fillna(0)

    pd.set_option("display.width", 250)
    print("=== Secondary analysis (excluding columnar cells) ===")
    print(sec.to_string(index=False))
    print("\n=== Internal vs external ===")
    print(ie.to_string(index=False))
    print("\n=== Rank agreement (interpret cautiously: only", len(ie), "architectures) ===")
    print(rank.to_string(index=False))
    for df, name in [(sec, "table_secondary_no_columnar.csv"), (ie, "table_internal_vs_external.csv"),
                     (rank, "table_rank_agreement.csv"), (mp, "table_class_mapping_all_archs.csv")]:
        out(df, name)
    print(f"\nTables saved to {HERLEV_DIR}")


if __name__ == "__main__":
    main()
