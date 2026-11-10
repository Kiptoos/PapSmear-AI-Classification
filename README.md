# PapSmear-AI-Classification

Code, fold assignments, out-of-fold predictions and external-evaluation results for:

> Kiptoo S, Nderu L, Mutanu L. *Quantifying source-image leakage and cross-dataset generalisation in cervical cell classification: a comparative evaluation of convolutional neural network and transformer models.* PeerJ Computer Science (under revision, 2026).

SIPaKMeD's 4,049 cropped-cell images come from only 966 source cluster images, so image-level random splitting can place cells from the same source image in both training and test data. The study evaluates six ImageNet-pretrained backbones (four CNNs and two vision transformers) at three levels:

1. **Image-level (naive)** five-fold cross-validation, the protocol commonly used on SIPaKMeD.
2. **Source-disjoint** five-fold cross-validation, grouped by source image (both repeated five times).
3. **External evaluation** on the independent Herlev dataset, without retraining.

It quantifies the Source-image Leakage Inflation (SLI), tests its mechanism (negative control and dose-response), measures the effect of cell clustering on statistical precision, and compares the leakage effect with the loss of performance on Herlev. Every reported number, table and figure is computed from released predictions.

## Repository structure

```
PapSmear-AI-Classification/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── run_all.bat                      Windows: runs steps 1-3 for all backbones
├── src/
│   ├── common.py                    configuration, dataset/group parsing, splits, models
│   ├── sort_dataset.py              step 0: sort the SIPaKMeD download into dataset/<Class>/
│   ├── extract_features.py          step 1: frozen-backbone features, once per backbone
│   ├── run_cv.py                    step 2: repeated 5-fold CV, naive and source-disjoint
│   ├── analyze.py                   step 3: main tables, figures and statistics
│   ├── gradcam.py                   step 4: Grad-CAM on held-out cells
│   ├── prepare_oof.py               step 5: combine OOF predictions for the bootstrap
│   ├── sli_bootstrap.py             step 6: paired cluster bootstrap (class-level SLI, calibration, ranking)
│   ├── leakage_mechanism.py         step 7: negative control, dose-response, cluster vs cell bootstrap
│   ├── herlev_eval.py               step 8: external evaluation on Herlev, per backbone
│   └── herlev_secondary.py          step 9: analysis excluding columnar cells, rank agreement
├── notebooks/
│   └── 00_data_verification.ipynb   class counts, filename parser check, source-image statistics
├── outputs/
│   ├── predictions/                 out-of-fold predictions, one file per backbone
│   ├── results/                     tables (CSV), fold composition, OOF prediction release file
│   │   ├── sli_bootstrap/           class-level SLI, calibration, ranking stability
│   │   ├── leakage_mechanism/       SLI by source-image size, dose-response, design effects
│   │   └── herlev/                  external evaluation: per-image predictions and tables
│   └── figures/                     figures used in the manuscript
└── legacy/
    ├── notebooks/                   notebooks of the original submission
    ├── gradcam/                     Grad-CAM images of the original submission
    └── results/                     results of the original submission (superseded)
```

Not included: the SIPaKMeD and Herlev images (`dataset/`, `raw/`, `data/`) and cached features (`outputs/features/`). See [Data](#data).

## Key results

### 1. Source-image leakage on SIPaKMeD

Five-class accuracy (%) under five-fold cross-validation repeated five times (mean ± SD, n = 4,049). SLI = naive minus source-disjoint accuracy, with source-image cluster-bootstrap 95% confidence intervals.

| Backbone | Naive | Source-disjoint | SLI (pp) | 95% CI |
|---|---|---|---|---|
| ResNet50 | 94.93 ± 0.21 | **94.07 ± 0.27** | 0.86 | 0.46–1.33 |
| BEiT-B/16 | 93.89 ± 0.21 | 92.68 ± 0.29 | 1.22 | 0.84–1.61 |
| EfficientNetB3 | 93.63 ± 0.16 | 92.69 ± 0.17 | 0.94 | 0.57–1.30 |
| DenseNet169 | 93.49 ± 0.27 | 92.36 ± 0.21 | 1.13 | 0.69–1.61 |
| ConvNeXt-Tiny | 92.53 ± 0.15 | 91.66 ± 0.25 | 0.87 | 0.43–1.41 |
| TinyViT-5M | 88.15 ± 0.30 | 86.97 ± 0.23 | 1.18 | 0.81–1.55 |

Under naive splitting, 91.6% of validation images share a source image with the training data. All six backbones show a positive SLI (p < 0.001). ResNet50 has the highest source-disjoint accuracy (macro-F1 94.06%, macro AUC 0.9947, ECE 0.019).

### 2. Mechanism and statistical precision

- **Negative control:** cells from single-cell source images, which have no siblings under either protocol, show no inflation (SLI −0.27 pp, 95% CI −0.69 to 0.12).
- **Dose-response:** the inflation rises with the number of sibling cells in the training data, to 1.69 pp for cells from source images with 11 or more cells; after adjustment for class, each additional sibling adds 0.10 pp (95% CI 0.04–0.20).
- **By class:** Parabasal, the most clustered class, is the only class with significant inflation for all six backbones.
- **Clustering:** cluster-bootstrap intervals for accuracy are 1.64–1.72 times wider than cell-level intervals (design effect 2.7–3.0); treating cells as independent makes confidence intervals about 40% too narrow.

### 3. External evaluation on Herlev

SIPaKMeD-trained models tested without retraining on Herlev (917 cells), abnormal versus non-abnormal, threshold 0.5. BA = balanced accuracy. The last column is a post hoc analysis excluding the 98 columnar cells, a cell type absent from SIPaKMeD.

| Backbone | Internal source-disjoint BA (%) | Herlev BA (%) | Herlev AUC | Herlev AUC, columnar excluded |
|---|---|---|---|---|
| ResNet50 | 96.01 | 69.92 | 0.717 | 0.928 |
| DenseNet169 | 95.46 | 70.51 | 0.718 | 0.936 |
| BEiT-B/16 | 95.77 | 64.03 | 0.762 | 0.911 |
| EfficientNetB3 | 95.33 | 58.83 | 0.620 | 0.756 |
| ConvNeXt-Tiny | 94.90 | 65.34 | 0.705 | 0.880 |
| TinyViT-5M | 92.16 | 65.70 | 0.692 | 0.963 |

The drop from internal to external balanced accuracy (25–37 pp) is 37 to 71 times larger than the leakage effect on the same task. All six backbones flag normal columnar cells as abnormal about as often as dysplastic cells, and internal rankings do not predict external performance (Spearman ρ = 0.20 for balanced accuracy; −0.03 for AUC with columnar cells excluded).

## Data

Neither dataset is redistributed here.

### SIPaKMeD

Download SIPaKMeD (Plissiti et al., 2018) from Kaggle
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

### Herlev

Download the new Pap-smear database (`smear2005.zip`) from the MDE-Lab, University of the Aegean: https://mde-lab.aegean.gr/?p=21. Unzip it and copy the seven class folders from `New database pictures` into `data/herlev/`:

```
data/herlev/
├── carcinoma_in_situ/       150
├── light_dysplastic/        182
├── moderate_dysplastic/     146
├── normal_columnar/          98
├── normal_intermediate/      70
├── normal_superficiel/       74
└── severe_dysplastic/       197    (917 cells in total)
```

Keep the folder names unchanged; the scripts use them to assign labels. Segmentation masks (files ending in `-d.bmp`) are skipped automatically. For external evaluation the classes are grouped as non-abnormal (superficial, intermediate, columnar) and abnormal (the four dysplastic classes); on SIPaKMeD the abnormal group is Koilocytotic and Dyskeratotic.

## Protocol

| Item | Setting |
|---|---|
| Backbones | EfficientNetB3 (300 px), ResNet50, DenseNet169, ConvNeXt-Tiny, TinyViT-5M, BEiT-B/16 (224 px); ImageNet-pretrained (TinyViT-5M and BEiT-B/16: ImageNet-21k → 1k via keras-cv-attention-models), frozen |
| Head | GAP → dropout → dense (512 swish + BN for EfficientNetB3; 256 ReLU otherwise) → softmax |
| Splits | `StratifiedKFold` (naive) vs `StratifiedGroupKFold` by source image; 5 folds × 5 repeats, seeds 42–46 |
| Training | Adam 1e-4, batch 16, fixed 15 epochs |
| Not used | hyperparameter search, early stopping, LR scheduling, checkpoint selection, augmentation, SMOTE, class weights |
| Uncertainty | paired cluster bootstrap over source images (5,000 resamples for the SLI) |
| External evaluation | head trained on all SIPaKMeD images with 5 seeds; P(abnormal) = P(Koilocytotic) + P(Dyskeratotic); fixed threshold 0.5; nothing tuned on Herlev; cell-level bootstrap (Herlev has no source IDs) |

Because backbones are frozen and no augmentation is used, features are extracted once per backbone and only the head is trained in each fold. This is equivalent to training the frozen-backbone network end to end.

## Reproducing the results

Python 3.11 with TensorFlow 2.15.0, keras-cv-attention-models 1.4.3, scikit-learn 1.4.0, SciPy, NumPy 1.26.4 and pandas 2.1.4. Install dependencies into a virtual environment:

```
pip install -r requirements.txt
```

### Internal evaluation (SIPaKMeD)

From the repository root:

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

### Leakage analyses

```
python src/prepare_oof.py
python src/sli_bootstrap.py --oof outputs/results/oof_all_for_bootstrap.csv --out outputs/results/sli_bootstrap
python src/leakage_mechanism.py
```

### External evaluation (Herlev)

Requires the SIPaKMeD features from step 1 and the Herlev data in `data/herlev/`.

```
for A in ResNet50 EfficientNetB3 DenseNet169 TinyViT-5M ConvNeXtTiny BEiT-B16; do
    python src/herlev_eval.py --arch $A
done
python src/herlev_eval.py --summarize
python src/herlev_secondary.py
```

On a CPU-only Intel Core i7 workstation, SIPaKMeD feature extraction takes about 4–33 minutes per backbone (ResNet50 fastest, ConvNeXt-Tiny slowest), cross-validation a few minutes per backbone, and Herlev evaluation about 1–10 minutes per backbone. The bootstrap and mechanism analyses take a few minutes in total.

## Outputs

### Internal evaluation

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

### Leakage analyses

| File | Content |
|---|---|
| `outputs/results/sli_bootstrap/sli_summary.csv` | SLI for accuracy, balanced accuracy and macro-F1 with paired cluster-bootstrap CIs |
| `outputs/results/sli_bootstrap/sli_per_class.csv` | class-level SLI (difference in recall) with CIs |
| `outputs/results/sli_bootstrap/calibration.csv` | ECE and Brier score per protocol; paired Brier difference with CIs |
| `outputs/results/sli_bootstrap/ranking_stability.csv` | backbone rankings under each protocol, Kendall's τ |
| `outputs/results/sli_bootstrap/pairwise_nadeau_bengio_source_disjoint.csv` | pairwise comparisons with the corrected resampled t-test |
| `outputs/results/leakage_mechanism/sli_by_source_size.csv` | SLI by number of cells per source image, including the single-cell negative control |
| `outputs/results/leakage_mechanism/dose_response_slope.csv` | class-adjusted slope of SLI on the number of training siblings |
| `outputs/results/leakage_mechanism/cluster_vs_image_bootstrap.csv` | cluster vs cell-level bootstrap intervals and design effects |

### External evaluation

| File | Content |
|---|---|
| `outputs/results/herlev/<backbone>_per_image.csv` | P(abnormal) for every Herlev cell and head seed |
| `outputs/results/herlev/<backbone>_summary.json` | internal and external binary metrics with CIs; flag rate per Herlev class |
| `outputs/results/herlev/<backbone>_herlev_vs_sipakmed_classes.csv` | SIPaKMeD class assigned to each Herlev class (first head seed) |
| `outputs/results/herlev/table_external_validation.csv` | internal naive, internal source-disjoint and Herlev results for all backbones |
| `outputs/results/herlev/table_herlev_flag_rate_per_class.csv` | % of cells classified abnormal per Herlev class |
| `outputs/results/herlev/table_secondary_no_columnar.csv` | primary and columnar-excluded results with CIs |
| `outputs/results/herlev/table_internal_vs_external.csv` | leakage gap and external drop per backbone |
| `outputs/results/herlev/table_rank_agreement.csv` | Spearman and Kendall agreement between internal and external rankings |
| `outputs/results/herlev/table_class_mapping_all_archs.csv` | Herlev-to-SIPaKMeD class assignments for all backbones |

## Legacy code

`legacy/` contains the notebooks, Grad-CAM images and result files from the original submission (June 2026). They used single train/validation splits, pixel-space SMOTE and a different training procedure, and they do **not** implement the protocol above. They are kept for transparency only; no result in the revised manuscript comes from them. The files in `legacy/results/` report the original, leakage-affected numbers and are superseded by `outputs/results/`.

## Citation

```bibtex
@article{kiptoo2026leakage,
  title   = {Quantifying source-image leakage and cross-dataset generalisation in cervical cell classification: a comparative evaluation of {convolutional neural network} and transformer models,
  author  = {Kiptoo, Stephen and Nderu, Lawrence and Mutanu, Leah},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Under revision}
}
```

Please also cite the datasets:

- Plissiti ME, Dimitrakopoulos P, Sfikas G, Nikou C, Krikoni O, Charchanti A. 2018. SIPaKMeD: a new dataset for feature and image based classification of normal and pathological cervical cells in Pap smear images. *IEEE ICIP 2018*, 3144–3148. https://doi.org/10.1109/ICIP.2018.8451588
- Jantzen J, Norup J, Dounias G, Bjerregaard B. 2005. Pap-smear benchmark data for pattern classification. In: *Proceedings of Nature-Inspired Smart Information Systems (NiSIS 2005)*, Albufeira, Portugal, 1–9.

## License

Code: MIT (see `LICENSE`). The SIPaKMeD and Herlev images are subject to their authors' terms and are not included.

## Contact

Stephen Kiptoo, stephen.kiptoo@jkuat.ac.ke / kiptoos@gmail.com (ORCID 0009-0008-1807-1989)
