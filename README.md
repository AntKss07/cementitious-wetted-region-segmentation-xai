# Automated Sorptivity & Wetted-Region Segmentation in Cementitious Materials

> **Interactive Research Workbench:** Launch the local scientific workbench with `python -m frontend.app.server --port 8000` or on Windows run `.\frontend\run.ps1 -Port 8000`. Provides live inference using the frozen best checkpoint (`evaluation/unetpp_best.pth`), physics-informed waterline post-processing, automated Wetted Area Ratio (WAR) analysis, and analytical Grad-CAM++ visual explainability.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%2F%2012.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning and Explainable AI (XAI) framework for **automated capillary water absorption (sorptivity) measurement** in concrete and cementitious materials. This repository provides standardized implementations of four modern semantic segmentation architectures, a rigorous multi-annotator consensus benchmark, automated physical **Wetted Area Ratio (WAR)** kinematic extraction, segmentation-specific **Grad-CAM++** interpretability, and a dedicated **Swiss Minimalist Scientific Workbench** web application.

---

## 📌 Table of Contents

- [Overview & Research Motivation](#-overview--research-motivation)
- [Interactive Research Workbench (Frontend)](#-interactive-research-workbench-frontend)
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
  - [1. Launch Interactive Web Workbench](#1-launch-interactive-web-workbench)
  - [2. Run Grad-CAM++ XAI Notebook](#2-run-grad-cam-xai-notebook)
  - [3. Run Model Training Pipelines](#3-run-model-training-pipelines)
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
│  Physical Domain Constraints & Kinematic Extraction    │
│  • Bottom Reservoir Contiguity (enforce_waterline_physics)
│  • Wetted Area Ratio (WAR) = Wet Pixels / Frame Pixels │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Explainable AI (Grad-CAM++)                           │
│  • Self-Consistent Mean Wet Logit Target               │
│  • Final Dense Decoder Layer: decoder.blocks.x_0_4     │
│  • Spatial Attribution Breakdown & Boundary Profiling  │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Interactive Research Workbench (Frontend)             │
│  • 4-Panel Synchronized Matrix                         │
│  • Interactive Split-Curtain Comparison Slider         │
│  • Live Telemetry & Error Maps (TP / FP / FN / TN)     │
└────────────────────────────────────────────────────────┘
```

---

## 🖥️ Interactive Research Workbench (Frontend)

The repository provides a complete web application built under the **Swiss Minimalist Scientific Instrument** aesthetic (governed by `ui-ux-pro-max`), strictly eliminating "vibe-coded" fluff, excessive glow, or low-contrast cards in favor of a clean, high-density laboratory interface.

```
frontend/
├── app/
│   ├── model.py            # Loads evaluation/unetpp_best.pth & normalizes tensors
│   ├── physics.py          # enforce_waterline_physics & compute_war
│   ├── gradcam.py          # Analytical Grad-CAM++ XAI engine
│   ├── dataset_catalog.py  # Indexes benchmark video frames & consensus masks
│   ├── pipeline.py         # End-to-end inference & comparison overlay generator
│   └── server.py           # Asynchronous aiohttp REST & static server
├── static/
│   ├── css/
│   │   ├── design-tokens.css # WCAG-compliant design tokens
│   │   └── workbench.css   # Swiss minimalist data-dense styling
│   ├── js/
│   │   ├── api.js          # Asynchronous backend client
│   │   └── app.js          # Interactive workbench & swipe slider controller
│   └── index.html          # Semantic workbench UI
├── requirements.txt        # Runtime dependencies
└── run.ps1                 # Windows launcher script
```

### Key Capabilities:
- **Specimen Ingestion**: One-click selection of untouched test benchmark frames (`video 21 - frame 040`, `video 1`, `video 24`) or drag-and-drop custom specimen uploads.
- **Physical Waterline Post-Processing**: Real-time toggle for bottom-reservoir contiguity and column monotonicity to filter out surface specularities and floating noise droplets.
- **Explainable AI**: Analytical Grad-CAM++ heatmaps computed across user-selectable target layers (`decoder.blocks.x_0_4` primary, `decoder.blocks.x_0_3`, or `encoder.layer4[-1]`) with adjustable colormaps (`Jet`, `Plasma`, `Inferno`, `Viridis`) and opacity sliders.
- **Interactive Comparison Slider**: Swipe slider enabling before/after inspection between Input Specimen and Physics-Enforced Mask, Input and Saliency Heatmap, or Ground Truth and Prediction.
- **Quantitative Telemetry**: Instant readout of Wetted Area Ratio (WAR %), pixel counts, estimated water front absorption height, floating noise filtered count, overlap metrics (Dice, IoU, Precision, Recall), and hardware latency breakdown.

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
│
├── frontend/                                 # Interactive Scientific Research Workbench
│   ├── app/
│   │   ├── __init__.py
│   │   ├── model.py                          # U-Net++ checkpoint loader & normalization
│   │   ├── physics.py                        # enforce_waterline_physics, compute_war
│   │   ├── gradcam.py                        # Analytical Grad-CAM++ XAI engine
│   │   ├── dataset_catalog.py                # Video sequence & consensus mask catalog
│   │   ├── pipeline.py                       # End-to-end inference & overlay generator
│   │   └── server.py                         # Asynchronous aiohttp REST & static server
│   ├── static/
│   │   ├── css/
│   │   │   ├── design-tokens.css             # WCAG AA/AAA design tokens
│   │   │   └── workbench.css                 # Swiss minimalist data-dense stylesheet
│   │   ├── js/
│   │   │   ├── api.js                        # Asynchronous backend client
│   │   │   └── app.js                        # Workbench controller & swipe comparison slider
│   │   └── index.html                        # Semantic laboratory workbench UI
│   ├── requirements.txt                      # Frontend runtime dependencies
│   └── run.ps1                               # Local hosting launcher script
│
├── Notebooks (Reproducible DL Pipeline)
│   ├── unetpp-res34.ipynb                    # U-Net++ (ResNet-34) Training & Evaluation (Winner)
│   ├── unetpp-res50.ipynb                    # U-Net++ (ResNet-50) Pipeline
│   ├── deeplabv3plus.ipynb                   # DeepLabV3+ (ResNet-50) Pipeline
│   ├── segformer_b1.ipynb                    # SegFormer-B1 (MiT-B1) Pipeline
│   ├── fpn1.ipynb                            # FPN (EfficientNet-B2) Pipeline
│   └── xai_gradcam_unetpp.ipynb              # 34-Cell Standalone Grad-CAM++ XAI Notebook
│
├── evaluation/                               # Benchmark Evaluation Checkpoints & Metrics
│   ├── unetpp_best.pth                       # Best U-Net++ checkpoint weights (Val IoU: 0.890)
│   ├── metrics_train_val.csv
│   └── metrics_test.csv
│
├── video-frame/                              # Video frames and multi-annotator ground truth
│   └── video-frame/
│       ├── frames/                           # Extracted frames across specimen videos
│       ├── consensus_masks/                  # Majority-vote consensus masks
│       └── annotations/                      # Annotator annotations
│
└── xai_gradcam_outputs/                      # Grad-CAM++ XAI Artifacts & Visualizations
    ├── tables/                               # Quantitative metrics and correlation summaries
    └── figures/                              # XAI overlays, boundary profiles, failure panels
```

---

## 👥 Dataset & Multi-Annotator Protocol

The dataset consists of high-resolution video recordings tracking capillary water absorption across 24 distinct cementitious specimens.

- **Split Strategy**: Video/sample-level split (prevents temporal data leakage).
  - **Train**: 17 videos (1,234 frames)
  - **Validation**: 4 videos (290 frames)
  - **Test**: 3 locked videos (`video 24`, `video 1`, `video 21`; 218 frames total)
- **Consensus Voting**:
  For each frame $x$, consensus mask $M$ is formed by pixel-wise voting across 4 independent annotators:
  $$M(i,j) = \mathbb{I}\left( \sum_{k=1}^4 M_k(i,j) \ge 2 \right)$$
- **Physical Parameter Formulation**:
  $$\text{WAR} = \frac{\sum_{(i,j)} M_{\text{wet}}(i,j)}{\text{Total Frame Pixels}}$$

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch albumentations opencv-python matplotlib pandas scikit-learn scipy aiohttp pillow
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Launch Interactive Web Workbench

To run the local scientific workbench application:

```bash
# Using Python directly:
python -m frontend.app.server --port 8000

# Or on Windows using PowerShell:
.\frontend\run.ps1 -Port 8000
```

Open **`http://127.0.0.1:8000`** in your browser.

- Select any benchmark frame (e.g. `Video 21 - Frame 040`, `Video 1 - Frame 001`, `Video 24 - Frame 064`) or upload an image.
- Review the 4-panel matrix: Input Specimen, Raw U-Net++ Output, Physics-Enforced Waterline Mask, and Grad-CAM++ Attribution Heatmap.
- Switch to the **Comparison Slider** to swipe between Input and Predicted Waterline.
- Export the complete quantitative analysis as JSON.

### 2. Run Grad-CAM++ XAI Notebook

Open and execute `xai_gradcam_unetpp.ipynb`:

```bash
jupyter notebook xai_gradcam_unetpp.ipynb
```

Running all cells will:
1. Load and freeze `evaluation/unetpp_best.pth`.
2. Verify test set metrics ($Dice = 0.965$, $IoU = 0.935$).
3. Generate publication-quality 6-panel Grad-CAM++ figures.
4. Calculate spatial attribution metrics across all 218 test frames.
5. Export correlation plots and failure analysis panels into `xai_gradcam_outputs/`.

### 3. Run Model Training Pipelines

Each architecture is packaged as an independent, reproducible notebook:
- `unetpp-res34.ipynb`: U-Net++ (ResNet-34)
- `unetpp-res50.ipynb`: U-Net++ (ResNet-50)
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
