"""
Step 4 - Experiment B: external evaluation on the Herlev dataset.

Uses the SAME frozen backbones, preprocessing and classification heads as the
SIPaKMeD CV (common.py), so nothing about the model changes between internal
and external evaluation.

    python src\\herlev_eval.py --arch ResNet50      (once per architecture)
    python src\\herlev_eval.py --summarize          (after all architectures)

What it does, per architecture
  1. Extracts frozen-backbone features for the 917 Herlev cells
     (cached in outputs/features/herlev_<arch>.npz, so it runs only once).
  2. Trains the classification head on ALL 4,049 SIPaKMeD cells with the same
     fixed hyperparameters as run_cv.py, using several seeds.
  3. Collapses the 5-class output to binary:
        P(abnormal) = P(Dyskeratotic) + P(Koilocytotic)
     (Metaplastic = benign and Parabasal / Superficial-Intermediate = normal
     count as negative), and evaluates on Herlev collapsed the same way:
        normal   = superficial, intermediate, columnar
        abnormal = mild, moderate, severe dysplasia, carcinoma in situ
  4. Computes the SAME binary metrics from the internal OOF predictions
     (naive and source-disjoint), so the paper can show three numbers side
     by side: internal naive -> internal source-disjoint -> external Herlev.

Outputs go to outputs/results/herlev/.
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import (ARCH_NAMES, INPUT_SIZE, CLASS_NAMES, CATEGORY_OF_CLASS,
                    FEATURE_DIR, RESULTS_DIR, BASE_SEED, EPOCHS, BATCH_SIZE,
                    LEARNING_RATE, load_image, preprocess, build_backbone,
                    global_pool, build_head, feature_path, prediction_path)

HERLEV_ROOT = os.path.join("data", "herlev")
OUT_DIR = os.path.join(RESULTS_DIR, "herlev")
ABN_IDX = [i for i, c in enumerate(CLASS_NAMES) if CATEGORY_OF_CLASS[c] == "Abnormal"]
PROB_COLS = [f"prob_{c}" for c in CLASS_NAMES]

# Herlev folder keyword -> (class name used in the paper, binary label)
HERLEV_RULES = [
    ("superfic", "Normal superficial", 0),
    ("intermed", "Normal intermediate", 0),
    ("columnar", "Normal columnar", 0),
    ("light", "Mild dysplasia", 1),
    ("moderate", "Moderate dysplasia", 1),
    ("severe", "Severe dysplasia", 1),
    ("carcinoma", "Carcinoma in situ", 1),
]
HERLEV_ORDER = [r[1] for r in HERLEV_RULES]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def collect_herlev(root=HERLEV_ROOT):
    paths, classes, ybin = [], [], []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        rule = next((r for r in HERLEV_RULES if r[0] in d.lower()), None)
        if rule is None:
            raise ValueError(f"Unrecognised Herlev folder: {d}")
        for f in sorted(os.listdir(full)):
            stem, ext = os.path.splitext(f)
            if ext.lower() != ".bmp" or re.search(r"-d$", stem, re.I):
                continue  # skip segmentation masks
            paths.append(os.path.join(full, f))
            classes.append(rule[1])
            ybin.append(rule[2])
    if len(paths) != 917:
        print(f"WARNING: expected 917 Herlev cells, found {len(paths)}")
    return paths, np.array(classes), np.array(ybin, dtype=np.int64)


def herlev_features(arch, paths, batch_size=16):
    """Frozen-backbone features for Herlev, cached after the first run."""
    cache = os.path.join(FEATURE_DIR, f"herlev_{arch}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=False)
        if list(d["paths"].astype(str)) == list(paths):
            print(f"Using cached Herlev features: {cache}")
            return d["features"]
        print("Cached Herlev features do not match current file list; re-extracting.")

    import gc
    import tensorflow as tf
    base = build_backbone(arch)
    size = INPUT_SIZE[arch]
    fwd = (tf.function(lambda x: base(x, training=False), reduce_retracing=True)
           if arch == "BEiT-B16" else (lambda x: base(x, training=False)))
    feats, t0 = [], time.time()
    for start in range(0, len(paths), batch_size):
        batch = np.stack([load_image(p, size) for p in paths[start:start + batch_size]])
        fmap = fwd(preprocess(arch, batch, base))
        feats.append(global_pool(np.asarray(fmap)))
        done = min(start + batch_size, len(paths))
        if (start // batch_size) % 20 == 0:
            gc.collect()
            print(f"  {done}/{len(paths)}  ({time.time() - t0:.0f}s)")
    feats = np.concatenate(feats).astype(np.float32)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    np.savez_compressed(cache, features=feats, paths=np.array(paths))
    print(f"Saved Herlev features {feats.shape} -> {cache}")
    del base
    tf.keras.backend.clear_session()
    return feats


def train_full_heads(arch, X, y, Xh, n_seeds):
    """Train the head on ALL SIPaKMeD features (same settings as run_cv.py)
    with n_seeds different seeds; return Herlev probabilities (n_seeds, n, 5)."""
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    out = []
    for s in range(n_seeds):
        tf.keras.utils.set_random_seed(BASE_SEED + 1000 + s)
        model = build_head(arch, X.shape[1])
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)
        out.append(model.predict(Xh, batch_size=256, verbose=0))
        print(f"  head seed {s + 1}/{n_seeds} trained")
        del model
        tf.keras.backend.clear_session()
    return np.stack(out)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y, score, thr=0.5):
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
    return {"sensitivity": sens, "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2, "f1_abnormal": f1, "auc": auc}


METRICS = ["sensitivity", "specificity", "balanced_accuracy", "f1_abnormal", "auc"]


def bootstrap(y_list, s_list, n_boot, seed, groups=None):
    """Percentile 95% CIs. y_list/s_list: one array per repeat (same image order).
    If groups is given, whole source images are resampled (cluster bootstrap)
    and the metric is averaged over repeats inside each draw."""
    rng = np.random.default_rng(seed)
    n = len(y_list[0])
    if groups is not None:
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    draws = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        per_rep = [binary_metrics(y[idx], s[idx]) for y, s in zip(y_list, s_list)]
        draws.append({k: np.nanmean([m[k] for m in per_rep]) for k in METRICS})
    d = pd.DataFrame(draws)
    return {k: [float(np.nanpercentile(d[k], 2.5)), float(np.nanpercentile(d[k], 97.5))]
            for k in METRICS}


def internal_binary(arch, n_boot, seed):
    """Binary normal-vs-abnormal metrics from the SIPaKMeD OOF predictions."""
    df = pd.read_csv(prediction_path(arch))
    df["y_bin"] = df["true_label"].map(lambda i: int(i in ABN_IDX))
    df["score"] = df[[PROB_COLS[i] for i in ABN_IDX]].sum(1)
    res = {}
    for strategy, g in df.groupby("strategy"):
        reps = [r.sort_values("filename") for _, r in g.groupby("repeat")]
        y_list = [r["y_bin"].to_numpy() for r in reps]
        s_list = [r["score"].to_numpy() for r in reps]
        point = {k: float(np.nanmean([binary_metrics(y, s)[k]
                                       for y, s in zip(y_list, s_list)])) for k in METRICS}
        ci = bootstrap(y_list, s_list, n_boot, seed, groups=reps[0]["group"].to_numpy())
        res[strategy] = {"point": point, "ci": ci, "n_repeats": len(reps)}
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_arch(arch, n_seeds, n_boot):
    os.makedirs(OUT_DIR, exist_ok=True)
    paths, hclass, yh = collect_herlev()
    print(f"Herlev: {len(paths)} cells ({int((yh == 0).sum())} normal, {int(yh.sum())} abnormal)")

    d = np.load(feature_path(arch), allow_pickle=False)
    X, y = d["features"], d["labels"]
    print(f"SIPaKMeD features for {arch}: {X.shape}")

    Xh = herlev_features(arch, paths)
    if Xh.shape[1] != X.shape[1]:
        raise SystemExit(f"Feature size mismatch: SIPaKMeD {X.shape[1]} vs Herlev {Xh.shape[1]}")

    P = train_full_heads(arch, X, y, Xh, n_seeds)          # (seeds, n, 5)
    scores = P[:, :, ABN_IDX].sum(2)                        # (seeds, n)

    # Per-image predictions
    per_img = pd.DataFrame({"path": paths, "herlev_class": hclass, "y_binary": yh})
    for s in range(n_seeds):
        per_img[f"p_abnormal_seed{s}"] = scores[s]
    per_img["p_abnormal_mean"] = scores.mean(0)
    per_img["pred_5class_seed0"] = [CLASS_NAMES[i] for i in P[0].argmax(1)]
    per_img.to_csv(os.path.join(OUT_DIR, f"{arch}_per_image.csv"), index=False)

    # External metrics: mean over seeds, CI by image bootstrap
    # (Herlev has no source/patient identifiers, so cells are resampled individually)
    seed_metrics = [binary_metrics(yh, scores[s]) for s in range(n_seeds)]
    external = {
        "point": {k: float(np.mean([m[k] for m in seed_metrics])) for k in METRICS},
        "sd_across_seeds": {k: float(np.std([m[k] for m in seed_metrics])) for k in METRICS},
        "ci": bootstrap([yh] * n_seeds, list(scores), n_boot, BASE_SEED),
    }
    flag_rate = (per_img.assign(flag=(scores >= 0.5).mean(0))
                 .groupby("herlev_class")["flag"].mean() * 100).reindex(HERLEV_ORDER)
    # Which SIPaKMeD class each Herlev class is mapped to most often (seed 0)
    crosstab = pd.crosstab(per_img["herlev_class"], per_img["pred_5class_seed0"]) \
        .reindex(HERLEV_ORDER).fillna(0).astype(int)
    crosstab.to_csv(os.path.join(OUT_DIR, f"{arch}_herlev_vs_sipakmed_classes.csv"))

    internal = internal_binary(arch, n_boot, BASE_SEED)

    summary = {"arch": arch, "n_seeds": n_seeds, "external_herlev": external,
               "internal": internal, "flag_rate_percent": flag_rate.round(2).to_dict()}
    with open(os.path.join(OUT_DIR, f"{arch}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {arch}: normal vs abnormal, balanced accuracy / AUC ===")
    for k, v in internal.items():
        print(f"  internal {k:16s} BA={100 * v['point']['balanced_accuracy']:.2f}%  "
              f"AUC={v['point']['auc']:.3f}")
    e = external["point"]
    print(f"  external Herlev          BA={100 * e['balanced_accuracy']:.2f}%  AUC={e['auc']:.3f}  "
          f"(sens {100 * e['sensitivity']:.1f}%, spec {100 * e['specificity']:.1f}%)")
    print("\n% of Herlev cells flagged abnormal, per class:")
    print(flag_rate.round(1).to_string())
    print(f"\nSaved to {OUT_DIR}")


def summarize():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*_summary.json")))
    if not files:
        raise SystemExit("No summaries found; run --arch first.")
    rows, flags = [], []
    fmt = lambda p, ci, pct: (f"{100 * p:.2f} [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]" if pct
                              else f"{p:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    for f in files:
        s = json.load(open(f))
        row = {"Architecture": s["arch"]}
        for label, block in [("Internal naive", s["internal"].get("naive")),
                             ("Internal source-disjoint", s["internal"].get("source_disjoint")),
                             ("External Herlev", s["external_herlev"])]:
            if block is None:
                continue
            row[f"{label} BA (%)"] = fmt(block["point"]["balanced_accuracy"],
                                          block["ci"]["balanced_accuracy"], True)
            row[f"{label} AUC"] = fmt(block["point"]["auc"], block["ci"]["auc"], False)
        e = s["external_herlev"]["point"]
        row["Herlev sensitivity (%)"] = round(100 * e["sensitivity"], 2)
        row["Herlev specificity (%)"] = round(100 * e["specificity"], 2)
        rows.append(row)
        flags.append({"Architecture": s["arch"], **s["flag_rate_percent"]})
    t1, t2 = pd.DataFrame(rows), pd.DataFrame(flags)
    t1.to_csv(os.path.join(OUT_DIR, "table_external_validation.csv"), index=False)
    t2.to_csv(os.path.join(OUT_DIR, "table_herlev_flag_rate_per_class.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t1.to_string(index=False))
    print("\n% flagged abnormal per Herlev class:")
    print(t2.to_string(index=False))
    print(f"\nTables saved to {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCH_NAMES)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--seeds", type=int, default=5, help="head-training seeds on full SIPaKMeD")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if args.summarize:
        summarize()
    elif args.arch:
        run_arch(args.arch, args.seeds, args.n_boot)
    else:
        ap.error("give --arch NAME or --summarize")


if __name__ == "__main__":
    main()
