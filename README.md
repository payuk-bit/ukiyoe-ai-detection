# Detecting AI-Generated Ukiyo-e

BSc Thesis — Tilburg University, 2026

Classifying AI-generated Ukiyo-e art from human-created woodblock prints using EfficientNet-B0 with integrated FFT and Gabor filter layers.

## Files
- `dataset.py` — Data loading, preprocessing, and augmentation
- `Effecienet.py` — Experiment 1: EfficientNet-B0 baseline
- `train_fft.py` — Experiment 2: EfficientNet-B0 + FFT layer
- `train_gabor.py` — Experiment 3: EfficientNet-B0 + Gabor layer
- `robustness_test.py` — Robustness evaluation under distribution shifts
- `shap_analysis.py` — SHAP explainability analysis

## Dataset
AI-ArtBench (Ukiyo-e subset): https://www.kaggle.com/datasets/ravidussilva/real-ai-art

## Requirements
Requirements

Python 3.11
PyTorch
torchvision
scikit-learn
matplotlib
numpy
shap

pip install torch torchvision scikit-learn matplotlib numpy shap

