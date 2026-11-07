"""
Step 1 - extract frozen-backbone features once per architecture.

    python src\\extract_features.py --arch ResNet50

Output: outputs/features/<arch>.npz  (features, labels, groups, filenames)
"""
import argparse
import os
import time
import numpy as np

from common import (ARCH_NAMES, INPUT_SIZE, FEATURE_DIR, collect_dataset,
                    load_image, preprocess, build_backbone, global_pool, feature_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCH_NAMES)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    paths, labels, groups, fnames = collect_dataset()
    print(f"{len(paths)} images, {len(set(groups))} source images")

    import gc
    import tensorflow as tf
    base = build_backbone(args.arch)
    size = INPUT_SIZE[args.arch]
    # BEiT-B16: compile the forward pass once; repeated eager calls leaked memory on CPU
    fwd = (tf.function(lambda x: base(x, training=False), reduce_retracing=True)
           if args.arch == "BEiT-B16" else (lambda x: base(x, training=False)))
    feats, t0 = [], time.time()
    for start in range(0, len(paths), args.batch_size):
        batch = np.stack([load_image(p, size) for p in paths[start:start + args.batch_size]])
        fmap = fwd(preprocess(args.arch, batch, base))
        feats.append(global_pool(fmap.numpy()))
        done = min(start + args.batch_size, len(paths))
        if (start // args.batch_size) % 20 == 0:
            gc.collect()
        if (start // args.batch_size) % 20 == 0 or done == len(paths):
            print(f"  {done}/{len(paths)}  ({time.time() - t0:.0f}s)")

    feats = np.concatenate(feats).astype(np.float32)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    np.savez_compressed(feature_path(args.arch), features=feats, labels=labels,
                        groups=np.array(groups), filenames=np.array(fnames))
    print(f"Saved {feats.shape} -> {feature_path(args.arch)}")


if __name__ == "__main__":
    main()
