# Brain Tumor Classification — 4-Class CNN on MRI Scans

A Convolutional Neural Network that classifies brain MRI scans into four tumor categories using a custom CNN trained on 5,600+ images with data augmentation and early stopping.

**72% test accuracy across 4 classes — 1,600 held-out test images**

---

## Overview

Accurate classification of brain tumor types from MRI scans is a critical step in medical diagnosis and treatment planning. This project builds a multi-class CNN classifier that distinguishes between four categories, trained on a balanced dataset with rigorous train/test separation.

---

## Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma | 0.84 | 0.46 | 0.60 |
| Meningioma | 0.63 | 0.45 | 0.52 |
| No Tumor | 0.65 | 1.00 | 0.79 |
| Pituitary | 0.80 | 0.97 | 0.87 |
| **Overall** | **0.73** | **0.72** | **0.70** |

**Test Accuracy: 71.94%** — evaluated on 1,600 held-out images (400 per class)

---

## Architecture

- **Conv2D (32 filters, 3×3, ReLU)** → MaxPooling
- **Conv2D (64 filters, 3×3, ReLU)** → MaxPooling
- **Conv2D (128 filters, 3×3, ReLU)** → MaxPooling
- **Flatten** → Dense (256, ReLU) → Dropout (0.5) → Dense (4, Softmax)

Loss: `sparse_categorical_crossentropy` | Optimizer: Adam

---

## Data Preprocessing

- Images resized to 150×150 pixels
- Pixel values normalized to [0, 1]
- Data augmentation: rotation, width/height shift, zoom, horizontal flip
- Balanced dataset: 400 test images per class

---

## Training

- Up to 20 epochs with `EarlyStopping` (restores best weights)
- `ReduceLROnPlateau` adjusts learning rate on plateau
- Separate training and test directories — no data leakage

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## How to Run

1. Open `brain_tumor_fixed.ipynb` in Google Colab
2. Mount Google Drive
3. Place training data in `MyDrive/ml/Training/` with subfolders: `glioma/`, `meningioma/`, `notumor/`, `pituitary/`
4. Place test data in `MyDrive/ml/Testing/` with the same structure
5. Run all cells in order

---

## Real-World Relevance

Brain tumor classification from MRI is an active area of medical AI research. Key challenges this project addresses:

- Distinguishing visually similar tumor types (glioma vs. meningioma)
- Avoiding false negatives on the no-tumor class (achieved 100% recall)
- Generalizing across image variability with data augmentation

## Next Steps

- Transfer learning with ResNet50 or EfficientNet for higher accuracy
- Class-weighted loss to improve meningioma recall
- Larger input resolution (224×224) for finer feature extraction

---

*Nicholas Menard*
