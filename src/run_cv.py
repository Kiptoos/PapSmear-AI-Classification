"""
Step 2 - repeated 5-fold CV of the classification head on cached features,
under both splitting strategies.

    python src\\run_cv.py --arch ResNet50 --repeats 5

Every image receives exactly one out-of-fold probability vector per
(strategy, repeat). Output: outputs/predictions/<arch>_oof.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

from common import (ARCH_NAMES, CLASS_NAMES, STRATEGIES, BASE_SEED, EPOCHS, BATCH_SIZE,
                    LEARNING_RATE, PRED_DIR, make_splits, build_head,
                    feature_path, prediction_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCH_NAMES)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    d = np.load(feature_path(args.arch), allow_pickle=False)
    X, y = d["features"], d["labels"]
    groups, fnames = d["groups"].astype(str), d["filenames"].astype(str)
    print(f"{args.arch}: features {X.shape}")

    rows = []
    for strategy in STRATEGIES:
        for r in range(args.repeats):
            seed = BASE_SEED + r
            splits = make_splits(y, groups, strategy, seed)
            seen = np.zeros(len(y), dtype=int)
            for fold, (tr, va) in enumerate(splits):
                tf.keras.utils.set_random_seed(seed * 100 + fold)
                model = build_head(args.arch, X.shape[1])
                model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                model.fit(X[tr], y[tr], epochs=EPOCHS, batch_size=BATCH_SIZE,
                          shuffle=True, verbose=0)
                prob = model.predict(X[va], batch_size=256, verbose=0)
                seen[va] += 1
                acc = float((prob.argmax(1) == y[va]).mean())
                print(f"  {strategy:15s} repeat {r + 1}/{args.repeats} fold {fold + 1}: "
                      f"n_val={len(va)}  acc={acc:.4f}")
                for j, i in enumerate(va):
                    row = {"arch": args.arch, "strategy": strategy, "repeat": r, "seed": seed,
                           "fold": fold, "filename": fnames[i], "group": groups[i],
                           "true_label": int(y[i])}
                    for k, c in enumerate(CLASS_NAMES):
                        row[f"prob_{c}"] = float(prob[j, k])
                    rows.append(row)
                del model
                tf.keras.backend.clear_session()
            assert (seen == 1).all(), "every image must be predicted exactly once per repeat"

    os.makedirs(PRED_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(prediction_path(args.arch), index=False)
    print(f"Saved -> {prediction_path(args.arch)}")


if __name__ == "__main__":
    main()
