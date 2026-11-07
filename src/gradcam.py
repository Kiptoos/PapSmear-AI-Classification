"""
Step 4 (optional) - Grad-CAM on HELD-OUT images, with true and predicted labels.

    python src\\gradcam.py --arch ResNet50 --per-class 2

Uses repeat 1 / fold 1 of the source-disjoint protocol: the head is trained on
that fold's training images only, and Grad-CAM is computed for validation
images from source images never seen in training. For each class it shows
`per-class` correctly classified cells plus up to `n-errors` misclassified
cells overall, so the figure is not cherry-picked to successes.
Grad-CAM is taken at the backbone's final feature map (frozen backbone,
GAP + trained head), and is presented as a qualitative illustration only.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (ARCH_NAMES, CLASS_NAMES, INPUT_SIZE, BASE_SEED, EPOCHS, BATCH_SIZE,
                    LEARNING_RATE, DATASET_ROOT, FIG_DIR, make_splits, build_head,
                    build_backbone, preprocess, load_image, feature_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCH_NAMES)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--n-errors", type=int, default=3)
    args = ap.parse_args()

    import tensorflow as tf
    d = np.load(feature_path(args.arch), allow_pickle=False)
    X, y = d["features"], d["labels"]
    groups, fnames = d["groups"].astype(str), d["filenames"].astype(str)

    tr, va = make_splits(y, groups, "source_disjoint", BASE_SEED)[0]
    tf.keras.utils.set_random_seed(BASE_SEED * 100)
    head = build_head(args.arch, X.shape[1])
    head.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                 loss="sparse_categorical_crossentropy")
    head.fit(X[tr], y[tr], epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)
    prob = head.predict(X[va], verbose=0)
    pred = prob.argmax(1)

    rng = np.random.default_rng(BASE_SEED)
    chosen = []
    for k in range(len(CLASS_NAMES)):
        ok = va[(y[va] == k) & (pred == k)]
        chosen += [(i, True) for i in rng.choice(ok, min(args.per_class, len(ok)), replace=False)]
    wrong = va[pred != y[va]]
    if len(wrong):
        chosen += [(i, False) for i in rng.choice(wrong, min(args.n_errors, len(wrong)), replace=False)]
    pos = {i: j for j, i in enumerate(va)}

    base = build_backbone(args.arch)
    size = INPUT_SIZE[args.arch]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 3 * len(chosen)))
    for row, (i, is_ok) in enumerate(chosen):
        img = load_image(os.path.join(DATASET_ROOT, fnames[i]), size)
        fmap = base(preprocess(args.arch, img[None]), training=False)
        if len(fmap.shape) != 4:
            raise SystemExit(f"{args.arch} does not return a spatial feature map; Grad-CAM not applicable.")
        c = int(pred[pos[i]])
        with tf.GradientTape() as tape:
            tape.watch(fmap)
            out = head(tf.reduce_mean(fmap, axis=[1, 2]), training=False)
            score = out[:, c]
        grads = tape.gradient(score, fmap)
        w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(w * fmap, axis=-1))[0].numpy()
        cam = cam / (cam.max() + 1e-8)
        cam = tf.image.resize(cam[..., None], (size, size), method="bilinear").numpy()[..., 0]

        title = (f"True: {CLASS_NAMES[y[i]]}\nPred: {CLASS_NAMES[c]} "
                 f"(p = {prob[pos[i], c]:.2f}) - {'correct' if is_ok else 'ERROR'}")
        axes[row, 0].imshow(img.astype(np.uint8)); axes[row, 0].set_title(title, fontsize=8)
        axes[row, 1].imshow(img.astype(np.uint8)); axes[row, 1].imshow(cam, cmap="jet", alpha=0.45)
        axes[row, 1].set_title("Grad-CAM overlay", fontsize=8)
        for a in axes[row]:
            a.axis("off")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"fig_gradcam_{args.arch}.png")
    fig.savefig(out, dpi=200)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
