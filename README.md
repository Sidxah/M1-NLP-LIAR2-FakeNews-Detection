# 🔍 Fake News Detection with LIAR2 — A Multimodal NLP Approach

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** Sid Ahmed BOUAMAMA & Aya MESSAOUDI  
**Course:** Hands-on Natural Language Processing — Université Paris-Saclay, M1 2025-2026  
**Date:** February 2026

---

## Overview

This project explores **fake news detection** as a multimodal NLP problem, demonstrating that **text alone is insufficient** and that **structured metadata** (speaker credit history) is the dominant predictive signal.

We progressively build from course-taught techniques (TF-IDF, Word2Vec, BERT fine-tuning) to research-inspired methods (Late Fusion, Focal Loss, Progressive Unfreezing), showing that each technique has strengths and limitations.

### Key Finding

> **Word2Vec + Random Forest (2013) beats DeBERTa-v3 (2021) by +7 F1 points.**
>
> This is because the transformer can only see text — and a 18-word political statement doesn't contain enough information to determine its veracity. The real signal is in the **speaker's credit history** (how many times they've lied before), which transformers can't process natively (the tokenizer splits "243" into sub-word tokens, losing numerical meaning). Our **Late Fusion architecture** solves this by giving the transformer the text and an MLP the numbers.

---

## Dataset: LIAR2

| Property | Value |
|----------|-------|
| **Source** | [PolitiFact](https://www.politifact.com/) via [HuggingFace](https://huggingface.co/datasets/chengxuphd/liar2) |
| **Size** | ~23,000 political statements |
| **Statement length** | ~18 words (mean) |
| **Original labels** | 6-way: pants-fire, false, barely-true, half-true, mostly-true, true |
| **Our labels** | 3-way: False, Mixed, True |
| **Split** | Train: 18,369 / Val: 2,297 / Test: 2,296 |
| **Key features** | statement, speaker, party, 6× credit history counts, context, justification |
| **SOTA** | FDHN (Xu & Kechadi, 2024) = 0.712 F1 macro (6-way enhanced) |

**Why LIAR2 and not ISOT?** ISOT achieves 95%+ accuracy because the model learns journalistic *style* (Reuters vs. random blogs), not *veracity*. LIAR2 forces the model to reason about factual content — a much harder and more realistic benchmark.

**Citation:** Wang, W.Y. (2017). *"Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection."* ACL 2017.

---

## Approach & Results

### Progression: Simple → Complex

```
TF-IDF (course) → Word2Vec (course) → Transformers (course) → Late Fusion (research)
     ↓                   ↓                     ↓                       ↓
 Text features     Dense embeddings      Deep semantics       Text + Metadata combined
   ~0.54 F1          ~0.49 F1              ~0.59 F1              ~0.70 F1
```

### Results Summary

| Category | Model | F1 Macro | MCC |
|----------|-------|----------|-----|
| **Fusion** | DeBERTa-v3 Fusion (Focal+LS) | **0.697** | 0.553 |
| **AutoML** | AutoGluon Multimodal | 0.683 | 0.539 |
| **Fusion** | RoBERTa Fusion (Focal+LS) | 0.681 | 0.534 |
| **Ensemble** | Final Ensemble | 0.677 | 0.528 |
| **Classical** | LR Metadata-Only V2 | 0.669 | 0.520 |
| **Classical** | Word2Vec + Metadata V1 | 0.664 | 0.513 |
| **Classical** | TF-IDF + Metadata V1 | 0.642 | 0.482 |
| **Transformer** | DeBERTa-v3 Full Fine-Tune | 0.590 | 0.403 |
| **Transformer** | RoBERTa Full Fine-Tune | 0.578 | 0.384 |
| **Transformer** | RoBERTa LoRA (r=16) | 0.573 | 0.374 |
| **Transformer** | DeBERTa-v3 LoRA (r=16) | 0.569 | 0.368 |
| **Transformer** | RoBERTa Frozen + Head | 0.544 | 0.333 |

### Techniques Used

| From Course | From Our Research |
|-------------|-------------------|
| TF-IDF, Word2Vec | Weighted Credit Score, Bayesian Credibility |
| BERT / RoBERTa / DeBERTa | Late Fusion architecture |
| Fine-tuning (frozen, full) | LoRA (Hu et al., 2021) |
| CrossEntropy loss | Focal Loss (Lin et al., 2017) |
| Adam optimizer | Label Smoothing (Szegedy et al., 2016) |
| | Differential Learning Rates |
| | Progressive Unfreezing (Howard & Ruder, 2018) |
| | AutoGluon AutoML benchmark |

---

## Project Structure

```
M1-NLP-LIAR2-FakeNews-Detection/
├── fake_news_liar2_project.ipynb   # Main notebook (all experiments)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── outputs/                         # Generated at runtime
    ├── figures/                     # All plots and confusion matrices
    ├── saved_models/                # Model checkpoints
    └── results.json                 # All metrics
```

---

## Quick Start

### Option 1: Kaggle (Recommended — free GPU)

1. Upload `fake_news_liar2_project.ipynb` to [Kaggle](https://www.kaggle.com/)
2. Enable **GPU** accelerator (Settings → Accelerator → GPU P100)
3. Enable **Internet** access
4. Run all cells

### Option 2: Local / Colab

```bash
git clone https://github.com/YOUR_USERNAME/M1-NLP-LIAR2-FakeNews-Detection.git
cd M1-NLP-LIAR2-FakeNews-Detection
pip install -r requirements.txt
jupyter notebook fake_news_liar2_project.ipynb
```

> **Requirements:** Python 3.10+, CUDA-compatible GPU (16GB VRAM recommended), ~8GB disk space for models.

---

## References

1. Wang, W.Y. (2017). *"Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection."* ACL 2017.
2. Xu, C. & Kechadi, M-T. (2024). *"FDHN: Fuzzy Deep Hybrid Network for Fake News Detection."* IEEE Access.
3. Hu, E.J. et al. (2021). *"LoRA: Low-Rank Adaptation of Large Language Models."* ICLR 2022.
4. Lin, T-Y. et al. (2017). *"Focal Loss for Dense Object Detection."* ICCV 2017.
5. Szegedy, C. et al. (2016). *"Rethinking the Inception Architecture for Computer Vision."* CVPR 2016.
6. Howard, J. & Ruder, S. (2018). *"Universal Language Model Fine-tuning for Text Classification."* ACL 2018.
7. He, P. et al. (2021). *"DeBERTa: Decoding-enhanced BERT with Disentangled Attention."* ICLR 2021.
8. Liu, Y. et al. (2019). *"RoBERTa: A Robustly Optimized BERT Pretraining Approach."*
9. Mikolov, T. et al. (2013). *"Efficient Estimation of Word Representations in Vector Space."*

---

## Acknowledgments

- **Professor Kim Gerdes & Nona Naderi** — Hands-on NLP course, Université Paris-Saclay
- **William Yang Wang** — for creating the original LIAR dataset
- **PolitiFact** — for the fact-checking data
- **Hugging Face** — for hosting the dataset and transformer models
