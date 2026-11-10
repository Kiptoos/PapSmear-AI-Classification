"""Figure 1: one sample cropped-cell image per SIPaKMeD class.

Same logic as notebooks/00_data_verification.ipynb, section 2 (first file of each
class, titled with class, filename and original size), laid out in two rows so the
saved PNG meets PeerJ's 900-3000 px limit on every side.

Run from the project root:  python src/make_figure1.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Work from the project root no matter where the script is started
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DATASET_ROOT = "dataset"
CLASS_NAMES = ["Dyskeratotic", "Koilocytotic", "Metaplastic",
               "Parabasal", "Superficial-Intermediate"]
OUT_PATH = os.path.join("outputs", "figures", "00_sample_images_per_class.png")

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.ravel()
for ax, class_name in zip(axes, CLASS_NAMES):
    files = sorted(f for f in os.listdir(os.path.join(DATASET_ROOT, class_name))
                   if f.lower().endswith(".bmp"))
    sample_file = files[0]
    img = Image.open(os.path.join(DATASET_ROOT, class_name, sample_file))
    ax.imshow(img)
    ax.set_title(f"{class_name}\n{sample_file}\n{img.size}")
    ax.axis("off")
axes[5].axis("off")  # empty sixth panel

plt.tight_layout(h_pad=3)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=280, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

w, h = Image.open(OUT_PATH).size
ok = 900 <= w <= 3000 and 900 <= h <= 3000
print(f"Saved {OUT_PATH}")
print(f"Size (px): ({w}, {h}) -> {'OK for PeerJ' if ok else 'OUTSIDE 900-3000, adjust dpi'}")
