```markdown
# PapSmear-AI-Classification

**Multi-Class Pap Smear Cell Classification Using EfficientNet with Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Explainability](#explainability)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 📖 Overview

This repository contains the complete code and implementation for the paper:

**"Multi-Class Pap Smear Cell Classification Using EfficientNet with Explainable AI: An AI Application for Cervical Cancer Screening"**

The project presents a comprehensive deep learning framework for automated Pap smear cell classification using transfer learning with EfficientNet architectures. The framework supports multi-granularity classification (binary, three-class, and five-class) and includes explainable AI capabilities through Grad-CAM visualizations.

**Key Contributions:**
- ✅ Five-fold cross-validation for robust performance evaluation
- ✅ Multi-granularity classification (binary, 3-class, 5-class)
- ✅ Grad-CAM explainability for clinical interpretability
- ✅ Confidence threshold analysis for deployment benchmarking
- ✅ Comprehensive architecture comparison (EfficientNetB0/B3, ResNet50, DenseNet169)

---

## 📊 Dataset

### SIPaKMeD Database

We use the **SIPaKMeD (Sivas Pathology Laboratory Pap Smear Image Database)**, a publicly available benchmark for Pap smear cell classification.

| Class | Images | Description |
|-------|--------|-------------|
| Dyskeratotic | 813 | Abnormal keratinization, associated with HPV |
| Koilocytotic | 825 | Perinuclear halos, pathognomonic for HPV |
| Metaplastic | 793 | Immature squamous cells |
| Parabasal | 787 | Small round cells, high N/C ratio |
| Superficial-Intermediate | 831 | Mature squamous cells (normal) |

**Dataset Source:** [SIPaKMeD Website](https://www.cs.uoi.gr/~marina/SIPaKMeD.html)

---

## ✨ Features

- **Deep Learning Models:** EfficientNetB0, EfficientNetB3, ResNet50, DenseNet169
- **Transfer Learning:** Feature extraction and fine-tuning phases
- **Data Augmentation:** Rotation, zoom, flip, brightness adjustment
- **Class Imbalance Handling:** SMOTE and class weights
- **Hyperparameter Optimization:** Grid search for optimal configuration
- **Evaluation Metrics:** Precision, Recall, F1-Score, Accuracy, AUC
- **Explainable AI:** Grad-CAM visualizations for model interpretability
- **Confidence Calibration:** Threshold analysis for deployment benchmarking

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.15 or higher
- CUDA-capable GPU (recommended but not required)

### Clone the Repository

```bash
git clone https://github.com/Kiptoos/PapSmear-AI-Classification.git
cd PapSmear-AI-Classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements File

```
tensorflow>=2.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scipy>=1.11.0
opencv-python>=4.8.0
pandas>=2.0.0
seaborn>=0.12.0
Pillow>=10.0.0
```

---

## 🚀 Usage

### 1. Dataset Preparation

```bash
# Download SIPaKMeD dataset
# Place images in the following structure:
data/
├── Dyskeratotic/
├── Koilocytotic/
├── Metaplastic/
├── Parabasal/
└── Superficial-Intermediate/
```

### 2. Data Preprocessing

```bash
python preprocess.py --data_dir data/ --output_dir processed_data/
```

### 3. Training Models

**Train EfficientNetB3:**
```bash
python train_efficientnet.py --model B3 --epochs 50 --batch_size 16 --lr 5e-4
```

**Train ResNet50:**
```bash
python train_resnet.py --epochs 50 --batch_size 16 --lr 5e-4
```

**Train DenseNet169:**
```bash
python train_densenet.py --epochs 50 --batch_size 16 --lr 5e-4
```

### 4. Five-Fold Cross-Validation

```bash
python cross_validate.py --model efficientnetb3 --folds 5
```

### 5. Evaluation

```bash
python evaluate.py --model_path models/efficientnetb3_best.h5 --test_data processed_data/test/
```

### 6. Generate Results

```bash
python generate_metrics.py --model models/efficientnetb3_best.h5 --output results/
```

### 7. Grad-CAM Visualization

```bash
python gradcam.py --model models/efficientnetb3_best.h5 --image sample.png --output heatmap.png
```

---

## 🧠 Model Architectures

| Model | Input Size | Parameters | ImageNet Top-1 Accuracy |
|-------|------------|------------|------------------------|
| EfficientNetB0 | 224×224 | 5.3M | 77.1% |
| EfficientNetB3 | 300×300 | 12.0M | 81.6% |
| ResNet50 | 224×224 | 25.6M | 74.9% |
| DenseNet169 | 224×224 | 14.3M | 76.2% |

---

## 📈 Results

### Best Performing Model: EfficientNetB3

| Metric | Value |
|--------|-------|
| **5-Fold CV Accuracy** | 93.65% ± 0.51% |
| **Binary Accuracy** | 98.51% |
| **3-Class Accuracy** | 91.58% |
| **5-Class Macro F1** | 0.91 |

### Architecture Comparison

| Architecture | Accuracy | Inference (ms) |
|--------------|----------|----------------|
| EfficientNetB0 | 85.40% | 42 |
| ResNet50 | 86.70% | 68 |
| DenseNet169 | 85.80% | 55 |
| **EfficientNetB3** | **93.65%** | **87** |

---

## 🔍 Explainability

Grad-CAM visualizations confirm that the model focuses on diagnostically relevant nuclear and perinuclear features, aligning with Bethesda System criteria.

![Grad-CAM Example](outputs/gradcam/gradcam_summary_grid.png)

---

## 📂 Repository Structure

```
PapSmear-AI-Classification/
├── README.md
├── LICENSE
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── efficientnet.py
│   │   ├── resnet.py
│   │   └── densenet.py
│   ├── training/
│   │   ├── train.py
│   │   └── cross_validate.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── metrics.py
│   └── explainability/
│       └── gradcam.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── results_analysis.ipynb
│
├── outputs/
│   ├── gradcam/
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── per_class_metrics.png
│
└── models/
    ├── efficientnetb3_best.h5
    └── checkpoints/
```

---

## 📝 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{kiptoo2024papsmear,
  title={Multi-Class Pap Smear Cell Classification Using EfficientNet with Explainable AI: An AI Application for Cervical Cancer Screening},
  author={Kiptoo, Stephen and Nderu, Lawrence and Mutanu, Leah},
  journal={PeerJ Computer Science},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Stephen Kiptoo**
- Email: kiptoos@gmail.com
- ORCID: 0009-0008-1807-1989

**Lawrence Nderu**
- Email: lnderu@jkuat.ac.ke

**Leah Mutanu**
- Email: lmutanu@usiu.ac.ke

---

## 🙏 Acknowledgments

- **SIPaKMeD Dataset:** Plissiti et al. (2018)
- **Bethesda System:** Nayar and Wilbur (2015)
- **TensorFlow Team:** For the deep learning framework

---

## 📚 References

1. World Health Organization (2024) Cervical cancer. WHO Fact Sheet.
2. Nayar R and Wilbur DC (Eds.) (2015) The Bethesda System for Reporting Cervical Cytology.
3. Tan M and Le QV (2019) EfficientNet: Rethinking model scaling. ICML.
4. He K, Zhang X, Ren S and Sun J (2016) Deep residual learning. CVPR.
5. Selvaraju RR et al. (2017) Grad-CAM: Visual explanations. ICCV.
```

---

## Quick Summary for PeerJ:

| Requirement | Status |
|:---|:---|
| Title - Project name | ✅ PapSmear-AI-Classification |
| Description | ✅ Complete overview |
| Dataset Information | ✅ SIPaKMeD details included |
| Code Information | ✅ Structure and usage explained |
| Usage Instructions | ✅ Step-by-step commands |
| Requirements | ✅ Dependencies listed |
| Methodology | ✅ Training and evaluation explained |
| Citations | ✅ Paper reference included |
| License | ✅ MIT License |

This README is professional, comprehensive, and meets all PeerJ requirements for reproducibility.
