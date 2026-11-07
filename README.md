# PapSmear-AI-Classification

Code, fold assignments and out-of-fold predictions for:

> Kiptoo S, Nderu L, Mutanu L. *Quantifying source-image leakage in Pap smear cell classification: a source-disjoint cross-validation benchmark of CNN and transformer backbones on SIPaKMeD.* PeerJ Computer Science (under revision, 2026).

The study measures how much image-level random splitting inflates accuracy on SIPaKMeD, whose 4,049 cropped-cell images come from only 966 source cluster images. Five ImageNet-pretrained backbones are evaluated under paired **naive** (image-level) and **source-disjoint** (grouped by source image) five-fold cross-validation, repeated five times. Every reported number, table and figure is computed from pooled out-of-fold predictions covering all 4,049 images.

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
└── legacy/                          notebooks of the original submission (not used for any result)
```

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
| Backbones | EfficientNetB3 (300 px), ResNet50, DenseNet169, ConvNeXt-Tiny, TinyViT-5M (224 px); ImageNet-pretrained, frozen |
| Head | GAP → dropout → dense (512 swish + BN for EfficientNetB3; 256 ReLU otherwise) → softmax |
| Splits | `StratifiedKFold` (naive) vs `StratifiedGroupKFold` by source image; 5 folds × 5 repeats, seeds 42–46 |
| Training | Adam 1e-4, batch 16, fixed 15 epochs |
| Not used | hyperparameter search, early stopping, LR scheduling, checkpoint selection, augmentation, SMOTE, class weights |

Because backbones are frozen and no augmentation is used, features are extracted once per backbone and only the head is trained in each fold. This is equivalent to training the frozen-backbone network end to end.

## Reproducing the results

Python 3.11. Install dependencies into a virtual environment:

```
pip install -r requirements.txt
```

Then, from the repository root:

```
python src/sort_dataset.py
run_all.bat                                   (Windows; or run the loop below)
python src/gradcam.py --arch <primary model>
```

Equivalent commands on any OS:

```
for A in ResNet50 EfficientNetB3 DenseNet169 TinyViT-5M ConvNeXtTiny; do
    python src/extract_features.py --arch $A
    python src/run_cv.py --arch $A --repeats 5
done
python src/analyze.py
```

On a CPU-only Intel Core i7 workstation, feature extraction takes about [[x]]–[[x]] minutes per backbone and cross-validation a few minutes per backbone.

## Outputs

| File | Content |
|---|---|
| `outputs/results/analysis_summary.txt` | all key numbers in one text file |
| `outputs/results/fold_composition.csv` | images, source images and class counts per fold; validation images sharing a source image with training |
| `outputs/results/table_main_metrics.csv` | accuracy, macro-F1, macro AUC, ECE, Brier, AURC per backbone and protocol |
| `outputs/results/table_sli.csv` | Source-image Leakage Inflation with cluster-bootstrap 95% CIs |
| `outputs/results/table_pairwise.csv` | pairwise backbone comparisons (source-disjoint), Holm-adjusted |
| `outputs/results/table_per_class_source_disjoint.csv` | per-class precision, recall, F1 |
| `outputs/results/oof_predictions_repeat1_all_models.csv` | one out-of-fold probability vector per image, backbone and protocol (first repetition), with fold and group IDs |
| `outputs/figures/` | confusion matrix, ROC curves, reliability diagram, risk-coverage curves, SLI plot, Grad-CAM |

## Legacy code

`legacy/` contains the notebooks from the original submission (June 2026). They used single train/validation splits, pixel-space SMOTE and a different training procedure, and they do **not** implement the protocol above. They are kept for transparency only; no result in the revised manuscript comes from them.

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

Please also cite the dataset: Plissiti ME, Dimitrakopoulos P, Sfikas G, Nikou C, Krikoni O, Charchanti A. 2018. SIPaKMeD: a new dataset for feature and image based classification of normal and pathological cervical cells in Pap smear images. *IEEE ICIP 2018*, 3144–3148.

## License

Code: MIT (see `LICENSE`). The SIPaKMeD images are subject to the dataset authors' terms and are not included.

## Contact

Stephen Kiptoo, stephen.kiptoo@jkuat.ac.ke (ORCID 0009-0008-1807-1989)
