# PapSmear-AI-Classification

Code, fold assignments and out-of-fold predictions for:

> Kiptoo S, Nderu L, Mutanu L. *Quantifying source-image leakage in Pap smear cell classification: a source-disjoint cross-validation benchmark of CNN and transformer backbones on SIPaKMeD.* PeerJ Computer Science (under revision, 2026).

The study measures how much image-level random splitting inflates accuracy on SIPaKMeD, whose 4,049 cropped-cell images come from only 966 source cluster images. Six ImageNet-pretrained backbones (four CNNs and two vision transformers) are evaluated under paired **naive** (image-level) and **source-disjoint** (grouped by source image) five-fold cross-validation, repeated five times. Every reported number, table and figure is computed from pooled out-of-fold predictions covering all 4,049 images.

## Repository structure

```
PapSmear-AI-Classification/
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.bat                      Windows: runs steps 1-3 for all backbones
├── src/
│   ├── common.py                    configuration, dataset/group parsing, splits, models
│   ├── sort_dataset.py              step 0: sort the SIPaKMeD download into dataset/<Class>/
│   ├── extract_features.py          step 1: frozen-backbone features, once per backbone
│   ├── run_cv.py                    step 2: repeated 5-fold CV, naive and source-disjoint
│   ├── analyze.py                   step 3: all tables, figures and statistics
│   └── gradcam.py                   step 4: Grad-CAM on held-out cells
├── notebooks/
│   └── 00_data_verification.ipynb   class counts, filename parser check, source-image statistics
├── outputs/
│   ├── results/                     tables (CSV), fold composition, OOF prediction release file
│   └── figures/                     figures used in the manuscript
└── legacy/
    ├── notebooks/                   notebooks of the original submission
    ├── gradcam/                     Grad-CAM images of the original submission
    └── results/                     results of the original submission (superseded)
```

## Key results

Five-class accuracy (%) under five-fold cross-validation repeated five times (mean ± SD, n = 4,049). SLI = Source-image Leakage Inflation (naive minus source-disjoint accuracy), with source-image cluster-bootstrap 95% confidence intervals.

| Backbone | Naive | Source-disjoint | SLI (pp) | 95% CI |
|---|---|---|---|---|
| ResNet50 | 94.93 ± 0.21 | **94.07 ± 0.27** | 0.86 | 0.46–1.33 |
| BEiT-B/16 | 93.89 ± 0.21 | 92.68 ± 0.29 | 1.22 | 0.84–1.61 |
| EfficientNetB3 | 93.63 ± 0.16 | 92.69 ± 0.17 | 0.94 | 0.57–1.30 |
| DenseNet169 | 93.49 ± 0.27 | 92.36 ± 0.21 | 1.13 | 0.69–1.61 |
| ConvNeXt-Tiny | 92.53 ± 0.15 | 91.66 ± 0.25 | 0.87 | 0.43–1.41 |
| TinyViT-5M | 88.15 ± 0.30 | 86.97 ± 0.23 | 1.18 | 0.81–1.55 |

Under naive splitting, 91.6% of validation images share a source image with the training data. All six backbones show a positive SLI (p < 0.001). ResNet50 has the highest source-disjoint accuracy (macro-F1 94.06%, macro AUC 0.9947, ECE 0.019).

## Data

SIPaKMeD (Plissiti et al., 2018) is not redistributed here. Download it from Kaggle
(https://www.kaggle.com/datasets/marinaeplissiti/sipakmed), for example with kagglehub:

```python
import kagglehub
path = kagglehub.dataset_download("marinaeplissiti/sipakmed")
print("Path to dataset files:", path)
```

or with the Kaggle CLI: `kaggle datasets download -d marinaeplissiti/sipakmed -p raw`.
Copy or unpack the class folders (`im_Dyskeratotic`, ..., `im_Superficial-Intermediate`) into `raw/sipakmed/`, then run:

```
python src/sort_dataset.py
```

This copies the cropped-cell `.bmp` files from each `CROPPED/` folder into `dataset/<Class>/`, keeps the original filenames, and checks the class counts (813 / 825 / 793 / 787 / 831; 4,049 total).

**Source-image groups.** Filenames have the form `{source_image_id}_{crop}.bmp`. Source-image numbering restarts within each class, so the group key is `<Class>__<source_image_id>`, giving 966 groups. Patient and slide identifiers are not distributed with SIPaKMeD; the source image is the highest independent unit available.

## Protocol

| Item | Setting |
|---|---|
| Backbones | EfficientNetB3 (300 px), ResNet50, DenseNet169, ConvNeXt-Tiny, TinyViT-5M, BEiT-B/16 (224 px); ImageNet-pretrained (TinyViT-5M and BEiT-B/16: ImageNet-21k → 1k via keras-cv-attention-models), frozen |
| Head | GAP → dropout → dense (512 swish + BN for EfficientNetB3; 256 ReLU otherwise) → softmax |
| Splits | `StratifiedKFold` (naive) vs `StratifiedGroupKFold` by source image; 5 folds × 5 repeats, seeds 42–46 |
| Training | Adam 1e-4, batch 16, fixed 15 epochs |
| Not used | hyperparameter search, early stopping, LR scheduling, checkpoint selection, augmentation, SMOTE, class weights |

Because backbones are frozen and no augmentation is used, features are extracted once per backbone and only the head is trained in each fold. This is equivalent to training the frozen-backbone network end to end.

## Reproducing the results

Python 3.11 with TensorFlow 2.15.0, keras-cv-attention-models 1.4.3, scikit-learn 1.4.0, NumPy 1.26.4 and pandas 2.1.4. Install dependencies into a virtual environment:

```
pip install -r requirements.txt
```

Then, from the repository root:

```
python src/sort_dataset.py
run_all.bat                                   (Windows; or run the loop below)
python src/gradcam.py --arch ResNet50
```

Equivalent commands on any OS:

```
for A in ResNet50 EfficientNetB3 DenseNet169 TinyViT-5M ConvNeXtTiny BEiT-B16; do
    python src/extract_features.py --arch $A
    python src/run_cv.py --arch $A --repeats 5
done
python src/analyze.py
```

On a CPU-only Intel Core i7 workstation, feature extraction takes about 4–33 minutes per backbone (ResNet50 fastest, ConvNeXt-Tiny slowest) and cross-validation a few minutes per backbone.

## Outputs

| File | Content |
|---|---|
| `outputs/results/analysis_summary.txt` | all key numbers in one text file |
| `outputs/results/fold_composition.csv` | images, source images and class counts per fold; validation images sharing a source image with training |
| `outputs/results/table_main_metrics.csv` | accuracy, macro-F1, macro AUC, ECE, Brier, AURC per backbone and protocol |
| `outputs/results/table_sli.csv` | Source-image Leakage Inflation with cluster-bootstrap 95% CIs |
| `outputs/results/table_pairwise.csv` | pairwise backbone comparisons (source-disjoint), Holm-adjusted |
| `outputs/results/table_per_class_source_disjoint.csv` | per-class precision, recall, F1 |
| `outputs/results/metrics_per_repeat.csv` | every metric for each backbone, protocol and repetition |
| `outputs/results/granularity_per_repeat.csv` | five-class, three-category and binary (abnormal vs non-abnormal) results |
| `outputs/results/selective_classification_per_repeat.csv` | accuracy at 90/80/70/60/50% coverage |
| `outputs/results/confusion_matrix_ResNet50_repeat1.csv` | pooled out-of-fold confusion matrix for the primary model |
| `outputs/results/oof_predictions_repeat1_all_models.csv` | one out-of-fold probability vector per image, backbone and protocol (first repetition), with fold and group IDs |
| `outputs/figures/` | confusion matrix, ROC curves, reliability diagram, risk-coverage curves, SLI plot, Grad-CAM |

## Legacy code

`legacy/` contains the notebooks, Grad-CAM images and result files from the original submission (June 2026). They used single train/validation splits, pixel-space SMOTE and a different training procedure, and they do **not** implement the protocol above. They are kept for transparency only; no result in the revised manuscript comes from them. The files in `legacy/results/` report the original, leakage-affected numbers and are superseded by `outputs/results/`.

## Citation

```bibtex
@article{kiptoo2026sipakmed,
  title   = {Quantifying source-image leakage in Pap smear cell classification: a source-disjoint cross-validation benchmark of CNN and transformer backbones on SIPaKMeD},
  author  = {Kiptoo, Stephen and Nderu, Lawrence and Mutanu, Leah},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Under revision}
}
```

Please also cite the dataset: Plissiti ME, Dimitrakopoulos P, Sfikas G, Nikou C, Krikoni O, Charchanti A. 2018. SIPaKMeD: a new dataset for feature and image based classification of normal and pathological cervical cells in Pap smear images. *IEEE ICIP 2018*, 3144–3148. https://www.cs.uoi.gr/~sfikas/sipakmed2018.pdf

## License

Code: MIT (see `LICENSE`). The SIPaKMeD images are subject to the dataset authors' terms and are not included.

## Contact

Stephen Kiptoo, stephen.kiptoo@jkuat.ac.ke/kiptoos@gmail.com (ORCID 0009-0008-1807-1989)
