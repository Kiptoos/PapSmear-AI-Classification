"""
Shared configuration for the PapSmear_AI v3 pipeline (PeerJ revision).

Design (answers Reviewer 2 ED1-ED2, Reviewer 3 2.3-2.4):
  * Backbones are ImageNet-pretrained and FROZEN, with no augmentation, so each
    image's pooled backbone feature vector is deterministic. Features are
    extracted once per architecture (extract_features.py); only the
    classification head is trained inside each CV fold (run_cv.py).
    This is mathematically equivalent to training the frozen-backbone model
    end-to-end, but orders of magnitude faster on CPU.
  * Grouping unit = source cluster image (the highest independent unit that
    SIPaKMeD exposes). Group ID is recovered from the filename
    {source_image_id}_{crop_number}.bmp, combined with the class folder
    because source-image numbering restarts within each class.
  * No hyperparameter search, no early stopping, no LR scheduling on the
    evaluation fold: fixed hyperparameters chosen a priori and a fixed number
    of epochs, so outer folds are untouched until prediction.
"""
import os
import re
import numpy as np

DATASET_ROOT = "dataset"
FEATURE_DIR = os.path.join("outputs", "features")
PRED_DIR = os.path.join("outputs", "predictions")
RESULTS_DIR = os.path.join("outputs", "results")
FIG_DIR = os.path.join("outputs", "figures")

CLASS_NAMES = [
    "Dyskeratotic", "Koilocytotic", "Metaplastic",
    "Parabasal", "Superficial-Intermediate",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# SIPaKMeD's own three categories (Plissiti et al., 2018).
# These are cell-morphology categories, NOT Bethesda diagnostic grades.
CATEGORY_OF_CLASS = {
    "Dyskeratotic": "Abnormal",
    "Koilocytotic": "Abnormal",
    "Metaplastic": "Benign",
    "Parabasal": "Normal",
    "Superficial-Intermediate": "Normal",
}
CATEGORIES = ["Normal", "Benign", "Abnormal"]

FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+)\.bmp$", re.IGNORECASE)

ARCH_NAMES = ["EfficientNetB3", "ResNet50", "DenseNet169", "ConvNeXtTiny", "TinyViT-5M", "BEiT-B16"]
INPUT_SIZE = {
    "EfficientNetB3": 300, "ResNet50": 224, "DenseNet169": 224,
    "ConvNeXtTiny": 224, "TinyViT-5M": 224, "BEiT-B16": 224,
}

STRATEGIES = ["naive", "source_disjoint"]
N_SPLITS = 5
BASE_SEED = 42

# Fixed, a-priori hyperparameters (identical across folds and strategies)
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 1e-4


def collect_dataset():
    """Return paths, labels (np.int64), group keys and filenames for all images."""
    paths, labels, groups, fnames = [], [], [], []
    for cls in CLASS_NAMES:
        cdir = os.path.join(DATASET_ROOT, cls)
        for fname in sorted(os.listdir(cdir)):
            if not fname.lower().endswith(".bmp"):
                continue
            m = FILENAME_PATTERN.match(fname)
            if m is None:
                raise ValueError(f"Unexpected filename: {cls}/{fname}")
            paths.append(os.path.join(cdir, fname))
            labels.append(CLASS_TO_IDX[cls])
            groups.append(f"{cls}__{m.group(1)}")
            fnames.append(f"{cls}/{fname}")
    return paths, np.array(labels, dtype=np.int64), groups, fnames


def make_splits(labels, groups, strategy, seed, n_splits=N_SPLITS):
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    X = np.zeros(len(labels))
    if strategy == "naive":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(X, labels))
    if strategy == "source_disjoint":
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(X, labels, groups))
    raise ValueError(strategy)


# ---------------- TensorFlow-dependent helpers (imported lazily) -------------

def load_image(path, size):
    import tensorflow as tf
    img = tf.keras.utils.load_img(path, target_size=(size, size))
    return tf.keras.utils.img_to_array(img)


def preprocess(arch, x, base=None):
    x = np.array(x, dtype=np.float32, copy=True)
    if arch == "BEiT-B16":
        # kecam models carry the preprocessing that matches their pretrained weights
        return base.preprocess_input(x)
    if arch == "EfficientNetB3":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif arch == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif arch == "DenseNet169":
        from tensorflow.keras.applications.densenet import preprocess_input
    elif arch == "ConvNeXtTiny":
        from tensorflow.keras.applications.convnext import preprocess_input
    elif arch == "TinyViT-5M":
        from tensorflow.keras.applications.imagenet_utils import preprocess_input
        return preprocess_input(x, mode="torch")
    else:
        raise ValueError(arch)
    return preprocess_input(x)


def build_backbone(arch):
    s = INPUT_SIZE[arch]
    shape = (s, s, 3)
    if arch == "TinyViT-5M":
        from keras_cv_attention_models import tinyvit
        base = tinyvit.TinyViT_5M(pretrained="imagenet21k-ft1k", input_shape=shape, num_classes=0)
    elif arch == "BEiT-B16":
        base = _build_beit_b16(shape)
    else:
        import tensorflow.keras.applications as apps
        ctor = {"EfficientNetB3": apps.EfficientNetB3, "ResNet50": apps.ResNet50,
                "DenseNet169": apps.DenseNet169, "ConvNeXtTiny": apps.ConvNeXtTiny}[arch]
        base = ctor(include_top=False, weights="imagenet", input_shape=shape)
    base.trainable = False
    return base


def _build_beit_b16(shape):
    """BEiT-B/16 (plain ViT-Base architecture; Bao et al., 2022) from
    keras-cv-attention-models, ImageNet-21k pretrained and ImageNet-1k fine-tuned.
    kecam prints a warning (not an error) and builds a RANDOM model when weights
    are missing, so its output is captured and the build is accepted only if
    kecam reports 'Load pretrained from'."""
    import io, contextlib
    from keras_cv_attention_models import beit
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        base = beit.BeitBasePatch16(pretrained="imagenet21k-ft1k", input_shape=shape, num_classes=0)
    out = buf.getvalue()
    print(out, end="")
    if "Load pretrained from" not in out:
        raise SystemExit("BEiT-B16 pretrained weights were NOT loaded; paste the messages above to Claude.")
    base.trainable = False
    print(">>>> BEiT-B16 pretrained weights VERIFIED: imagenet21k-ft1k (224 x 224)")
    return base


def global_pool(fmap):
    """Global average pooling for 4-D (B,H,W,C) or 3-D (B,T,C) outputs."""
    fmap = np.asarray(fmap)
    if fmap.ndim == 4:
        return fmap.mean(axis=(1, 2))
    if fmap.ndim == 3:
        return fmap.mean(axis=1)
    return fmap


def build_head(arch, feat_dim):
    """Same classification heads as the v2 experiments."""
    import tensorflow as tf
    from tensorflow.keras import layers
    inp = tf.keras.Input(shape=(feat_dim,))
    if arch == "EfficientNetB3":
        x = layers.Dropout(0.4)(inp)
        x = layers.Dense(512, activation="swish")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    else:
        x = layers.Dropout(0.3)(inp)
        x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def feature_path(arch):
    return os.path.join(FEATURE_DIR, f"{arch}.npz")


def prediction_path(arch):
    return os.path.join(PRED_DIR, f"{arch}_oof.csv")
