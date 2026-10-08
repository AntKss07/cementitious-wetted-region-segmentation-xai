# Automated Sorptivity & Wetted-Region Segmentation in Cementitious Materials

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%2F%2012.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning and Explainable AI (XAI) framework for **automated capillary water absorption (sorptivity) measurement** in concrete and cementitious materials. This repository provides standardized implementations of four modern semantic segmentation architectures, a rigorous multi-annotator consensus benchmark, automated physical **Wetted Area Ratio (WAR)** kinematic extraction, and segmentation-specific **Grad-CAM++** interpretability.

---

## 📌 Table of Contents

- [Overview & Research Motivation](#-overview--research-motivation)
- [Benchmark Results](#-benchmark-results)
- [Explainable AI (XAI) via Grad-CAM++](#-explainable-ai-xai-via-grad-cam)
  - [Mathematical Formulation for Segmentation](#mathematical-formulation-for-segmentation)
  - [Empirical Layer Selection](#empirical-layer-selection)
  - [Quantitative Spatial Attribution Diagnostics](#quantitative-spatial-attribution-diagnostics)
  - [Attribution Leakage vs. Physical WAR Error](#attribution-leakage-vs-physical-war-error)
- [Repository Structure](#-repository-structure)
- [Dataset & Multi-Annotator Protocol](#-dataset--multi-annotator-protocol)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
- [Citation & License](#-citation--license)

---

## 🔬 Overview & Research Motivation

Sorptivity characterizes the rate of water absorption by capillary suction into unsaturated concrete, acting as a primary indicator of durability, permeability, and service life. Traditional ASTM C1585 sorptivity testing relies on gravimetric mass measurements that fail to capture:
1. Multi-dimensional, non-uniform capillary ingress fronts;
2. Localized preferential absorption channels and micro-structural defects;
3. Real-time water front height kinematics.

This project implements computer-vision-based sorptivity testing via high-resolution video frame analysis ($448 \times 448$ RGB), comparing four segmentation architectures under strictly identical experimental conditions, followed by post-hoc explainability to verify that the selected model relies on physical capillary ingress rather than background apparatus artifacts.

```text
Input Specimen Image (448×448 RGB)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Multi-Architecture Benchmark (Identical Splits & Res) │
│  • U-Net++ (ResNet-34)      • DeepLabV3+ (ResNet-50)   │
│  • SegFormer-B1 (MiT-B1)    • FPN (EfficientNet-B2)    │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Selected Model: U-Net++ (ResNet-34)                   │
│  Dice = 0.965 | IoU = 0.934 | WAR R² = 0.998           │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Physical Parameter Estimation                         │
│  Wetted Area Ratio (WAR) = Wet Pixels / Specimen ROI   │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Explainable AI (Grad-CAM++)                           │
│  • Self-Consistent Mean Wet Logit Target               │
│  • Final Dense Decoder Layer: decoder.blocks.x_0_4     │
│  • Spatial Attribution Breakdown & Boundary Profiling  │
└────────────────────────────────────────────────────────┘
```

---

## 🏆 Benchmark Results

All four models were trained, validated, and evaluated on an identical **video-level split** (testing on `video 24`, `video 1`, and `video 21`; 218 test frames total) using ground-truth consensus masks derived from 4 independent human annotators ($\ge 2$ votes).

| Architecture | Encoder / Backbone | Params (M) | Test Dice ↑ | Test IoU ↑ | Test Precision ↑ | Test Recall ↑ | WAR MAE ↓ | WAR RMSE ↓ | WAR $R^2$ ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net++** 🥇 | **ResNet-34** | **26.1** | **0.9652** | **0.9346** | **0.9696** | 0.9608 | **0.0057** | **0.0075** | **0.9980** |
| **DeepLabV3+** | ResNet-50 | 39.8 | 0.9605 | 0.9254 | 0.9642 | 0.9575 | 0.0071 | 0.0092 | 0.9968 |
| **SegFormer** | MiT-B1 | 13.7 | 0.9592 | 0.9230 | 0.9631 | 0.9560 | 0.0078 | 0.0101 | 0.9961 |
| **FPN** | EfficientNet-B2 | 11.2 | 0.9521 | 0.9102 | 0.9510 | 0.9542 | 0.0094 | 0.0124 | 0.9942 |

**Key Findings**:
- **U-Net++ (ResNet-34)** achieved the best overall segmentation accuracy ($\text{Dice} = 0.9652$, $\text{IoU} = 0.9346$) and lowest physical WAR error ($\text{MAE} = 0.0057$, $R^2 = 0.9980$).
- Its nested skip connections effectively bridge fine-grained capillary boundary details with multi-scale semantic features, enabling sub-pixel accurate waterline tracking.

---

## 🔍 Explainable AI (XAI) via Grad-CAM++

Following the principle **"Train Once → Freeze → Explain"**, the top-performing U-Net++ checkpoint was frozen and interpreted using Grad-CAM++.

### Mathematical Formulation for Segmentation

Unlike classification where a single scalar class logit is backpropagated, semantic segmentation produces a spatial logit tensor $\mathbf{Z} \in \mathbb{R}^{B \times 1 \times H \times W}$. We define a self-consistent, scale-invariant scalar target $S$ representing the mean logit across the model's own predicted wetted region $\hat{M}_{\text{wet}}$:

$$S = \frac{\sum_{(i,j) \in \hat{M}_{\text{wet}}} Z_{i,j}}{|\hat{M}_{\text{wet}}|}$$

With fallback to top-$K$ activations ($K=100$) if $|\hat{M}_{\text{wet}}| = 0$. Higher-order weighting coefficients $\alpha_{i,j}^{kc}$ are calculated to weight feature map activations $A^k$:

$$\alpha_{i,j}^{kc} = \frac{\frac{\partial^2 S}{\partial (A_{i,j}^k)^2}}{2 \frac{\partial^2 S}{\partial (A_{i,j}^k)^2} + \sum_{a,b} A_{a,b}^k \frac{\partial^3 S}{\partial (A_{i,j}^k)^3} + \epsilon}$$

$$L_{\text{Grad-CAM++}} = \text{ReLU}\left(\sum_k \left[ \sum_{i,j} \alpha_{i,j}^{kc} \cdot \text{ReLU}\left(\frac{\partial S}{\partial A_{i,j}^k}\right) \right] A^k\right)$$

### Empirical Layer Selection

| Target Layer | Resolution | Channels | Spatial Localization | Waterline Alignment | Leakage to Background | Recommendation |
| :--- | :---: | :---: | :--- | :--- | :---: | :--- |
| `encoder.layer4` | $14 \times 14$ | 512 | Coarse semantic context | Severely blurred ($32\times$ upsampling) | $4.5\%$ | Coarse context only |
| `decoder.blocks.x_0_3` | $224 \times 224$ | 32 | Fine multi-scale | Good boundary fidelity | $0.8\%$ | Strong secondary |
| `decoder.blocks.x_0_4` | $448 \times 448$ | 16 | **Sub-pixel native** | **Sharpest waterline localization** | **0.11%** | **Primary Selected Layer** |

### Quantitative Spatial Attribution Diagnostics

Evaluated across all 218 test images to prove model soundness and absence of background shortcut learning:

| Cohort | Number of Frames | Mean ROI CAM % | Mean Wet CAM % | Mean Background Leakage % | Mean Boundary Band CAM % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Test Frames** | **218** | **99.89%** | **99.34%** | **0.11%** | **17.78%** |
| **Top 3 Segmentations** | 3 | **99.98%** | **99.78%** | **0.02%** | **16.54%** |
| **Lowest 3 Segmentations** | 3 | **97.30%** | **92.20%** | **2.70%** | **28.45%** |
| **Highest WAR Error** | 3 | **99.75%** | **98.85%** | **0.25%** | **18.90%** |

### Attribution Leakage vs. Physical WAR Error

- **Leakage vs. Segmentation Dice**: Strong negative correlation ($r = -0.7268$, $p = 4.29 \times 10^{-37}$). When the network incorrectly samples background pixels, segmentation accuracy drops.
- **Leakage vs. Absolute WAR Error**: Significant rank correlation (Spearman $\rho = +0.4633$, $p = 5.36 \times 10^{-13}$). Elevated WAR estimation errors are statistically linked to attribution spread outside the specimen ROI.

---

## 📂 Repository Structure

```text
├── .gitignore                                # Excludes checkpoints (>100MB), venvs, agents, caches
├── README.md                                 # Project documentation & benchmark overview
├── video_split_assignment.csv                # Video-level train/validation/test split assignment
├── build_xai_notebook.py                     # Programmatic builder for XAI notebook
│
├── Notebooks (Reproducible Pipeline)
│   ├── unetpp1.ipynb                         # U-Net++ (ResNet-34) Training & Evaluation (Winner)
│   ├── deeplabv3plus.ipynb                   # DeepLabV3+ (ResNet-50) Pipeline
│   ├── segformer_b1.ipynb                    # SegFormer-B1 (MiT-B1) Pipeline
│   ├── fpn1.ipynb                            # FPN (EfficientNet-B2) Pipeline
│   └── xai_gradcam_unetpp.ipynb              # 34-Cell Standalone Grad-CAM++ XAI Notebook
│
├── evaluation/                               # Benchmark Evaluation Metrics & Visualizations
│   ├── four_architecture_benchmark_comparison.csv
│   ├── unetpp/                               # U-Net++ metrics, predictions, WAR trajectories
│   ├── deeplabv3plus/                        # DeepLabV3+ evaluation outputs
│   ├── segformer_b1/                         # SegFormer evaluation outputs
│   └── fpn/                                  # FPN evaluation outputs
│
├── video-frame/                              # Video frames and multi-annotator ground truth
│   ├── video frames/                         # Extracted frames across 24 specimen videos
│   ├── annotator 1/                          # Individual binary masks from Annotator 1
│   ├── annotator 2/                          # Individual binary masks from Annotator 2
│   ├── annotator 3/                          # Individual binary masks from Annotator 3
│   └── annotator 4/                          # Individual binary masks from Annotator 4
│
└── xai_gradcam_outputs/                      # Grad-CAM++ XAI Artifacts & Visualizations
    ├── tables/
    │   ├── per_image_test_metrics.csv        # Metrics across all 218 test images
    │   ├── representative_cases.csv          # Identified cases across 8 quantitative criteria
    │   ├── xai_spatial_attribution_diagnostics.csv # Frame-by-frame attribution fractions
    │   └── xai_summary.json                  # Machine-readable summary statistics & correlations
    ├── figures/
    │   ├── annotator_agreement_comparison.png # 4-annotator agreement vs. Grad-CAM++ attribution
    │   ├── attribution_vs_war_error.png      # Statistical correlation scatter plots
    │   ├── boundary_analysis_cases.png       # Waterline overlays & vertical profile plots
    │   └── failure_analysis_detailed.png     # RGB, GT, Pred, Error Map, and CAM failure panels
    ├── layer_comparison/                     # Layer-wise comparison (encoder.layer4 vs decoder)
    ├── best_cases/                           # 6-panel XAI visualizations (top Dice)
    ├── median_cases/                         # 6-panel XAI visualizations (median Dice)
    ├── worst_cases/                          # 6-panel XAI visualizations (lowest Dice)
    └── war_error_cases/                      # 6-panel XAI visualizations (highest WAR error)
```

---

## 👥 Dataset & Multi-Annotator Protocol

The dataset consists of high-resolution video recordings tracking capillary water absorption across 24 distinct cementitious specimens.

- **Split Strategy**: Video/sample-level split (prevents temporal data leakage).
  - **Train**: 17 videos (1,234 frames)
  - **Validation**: 4 videos (290 frames)
  - **Test**: 3 locked videos (`video 24`, `video 1`, `video 21`; 218 frames)
- **Consensus Voting**:
  For each frame $x$, consensus mask $M$ is formed by pixel-wise voting across 4 independent annotators:
  $$M(i,j) = \mathbb{I}\left( \sum_{k=1}^4 M_k(i,j) \ge 2 \right)$$
- **Physical Parameter Formulation**:
  $$\text{WAR} = \frac{\sum_{(i,j) \in \text{ROI}} M(i,j)}{|\text{ROI}|}$$

---

## ⚙️ Installation & Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AntKss07/cementitious-wetted-region-segmentation-xai.git
cd cementitious-wetted-region-segmentation-xai
```

### 2. Set Up Python Environment

Python 3.10+ and a CUDA-compatible GPU (NVIDIA RTX 3060/4060 or better) are recommended.

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch albumentations opencv-python matplotlib pandas scikit-learn scipy jupyter
```

---

## 🚀 Quickstart & Reproduction Guide

### Reproduce Explainable AI (Grad-CAM++)

Launch Jupyter Notebook and open `xai_gradcam_unetpp.ipynb`:

```bash
jupyter notebook xai_gradcam_unetpp.ipynb
```

Running all cells will:
1. Load and freeze `evaluation/unetpp/unetpp_best.pth`.
2. Verify test set metrics ($Dice = 0.9648$, $WAR\ R^2 = 0.9980$).
3. Generate publication-quality 6-panel Grad-CAM++ figures.
4. Calculate spatial attribution metrics across all 218 test frames.
5. Export correlation plots and failure analysis panels into `xai_gradcam_outputs/`.

### Run Model Training Pipelines

Each architecture is packaged as an independent, reproducible notebook:
- `unetpp1.ipynb`: U-Net++ (ResNet-34)
- `deeplabv3plus.ipynb`: DeepLabV3+ (ResNet-50)
- `segformer_b1.ipynb`: SegFormer-B1 (MiT-B1)
- `fpn1.ipynb`: FPN (EfficientNet-B2)

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).

If you find this code or dataset useful in your research, please cite:

```bibtex
@article{antkss2026sorptivity,
  title={Automated Capillary Sorptivity Measurement and Wetted-Region Segmentation in Cementitious Materials Using Deep Semantic Segmentation and Explainable AI},
  author={Sri Sudharsanan, K. and Collaborators},
  year={2026}
}
```
