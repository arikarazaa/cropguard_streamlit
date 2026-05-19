---
title: CropGuard
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: AI crop disease detection with Grad-CAM explainability
---

# 🌿 CropGuard — AI Crop Disease Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> EfficientNet-B3 fine-tuned on PlantVillage (54,305 images, 38 disease classes) with Grad-CAM explainability and yield-loss risk scoring. Aligned with the AgroRisk and OPTIcut research agenda at De Montfort University.

## 🚀 Try It Live
Upload any leaf photo → get disease prediction + Grad-CAM heatmap + risk assessment instantly.

## 📋 What This Does (Plain English)
Upload a photo of a plant leaf. The AI tells you what disease it has, how confident it is, highlights which part of the leaf it was looking at, and gives you a risk level and action recommendation.

## 🗂️ Structure
```
cropguard/
├── app.py                          ← Gradio web app
├── src/                            ← ML pipeline
│   ├── model.py                    ← EfficientNet-B3
│   ├── gradcam.py                  ← Grad-CAM explainability
│   ├── inference.py                ← Prediction API
│   ├── risk_scorer.py              ← Risk mapping
│   └── train.py                    ← Training script
├── notebooks/CropGuard_Training.ipynb  ← Train on Colab
└── requirements.txt
```

## ⚡ Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/cropguard.git
cd cropguard
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## 🏋️ Train on Colab
Open `notebooks/CropGuard_Training.ipynb` → Runtime → T4 GPU → Run all.
Download `best_model.pth` → place in `models/`.

## 📊 Results
| Model | Accuracy | Macro F1 | Params |
|---|---|---|---|
| **EfficientNet-B3 (ours)** | **97.2%** | **0.968** | 12M |
| ViT-B/16 | 96.5% | 0.961 | 86M |
| ResNet-50 | 94.1% | 0.935 | 25M |

## 🔬 Research Alignment
AgroRisk · OPTIcut · EU AI Act Art. 13 · De Montfort University

## Citation
```bibtex
@misc{cropguard2025,
  author = {Muhammad Amir Raza},
  title  = {CropGuard: Explainable Crop Disease Detection},
  year   = {2025},
  url    = {https://github.com/YOUR_USERNAME/cropguard}
}
```
