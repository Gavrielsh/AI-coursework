# 🚬 Smoking Detection using CNN and Fine-Tuned ResNet18
Author: Gavriel Shalem
Deep Learning Course – 2025


## 📘 Project Overview
This project focuses on binary image classification — determining whether a person is smoking or not in a given image. The task was part of a university-level deep learning course and was divided into two main phases:
1. Training a custom Convolutional Neural Network (CNN) from scratch.
2. Fine-tuning a pretrained ResNet18 model using transfer learning.

The dataset includes both original (regular) and augmented image sets. The project evaluates different optimizers, learning rates, batch sizes, and augmentation strategies to find the most effective configuration.

---

## 🎯 Objectives
- Classify images into two categories: **smoking** vs **not smoking**.
- Implement a CNN from scratch and fine-tune a pretrained ResNet18 model.
- Analyze the effect of **data augmentation**, **optimizer**, and **hyperparameters** on model performance.
- Use **ClearML** to track experiments and compare performance across runs.

---

## 🧠 AI Concepts Applied
- **Convolutional Neural Networks (CNN)**
- **Transfer Learning & Fine-Tuning** (ResNet18)
- **Cross Entropy Loss**
- **Optimizers**: Adam, SGD with Momentum
- **Batch Normalization, Dropout, ReLU activations**
- **Data Augmentation**: Resize, Horizontal Flip, Rotation, Brightness/Contrast
- **Image Preprocessing**: Normalization with ImageNet statistics
- **Overfitting and Generalization**
- **Experiment Tracking** with ClearML
- **Evaluation Metrics**: Accuracy, Loss
- **Visual Error Analysis**: Displaying misclassified images

---

## 📂 Code Structure

### 🧾 `dl_2025_ex3_q1.py` – Part A: CNN From Scratch
This script trains a **custom-built CNN** with 3 convolutional layers, followed by 2 fully connected layers. It includes:
- Custom Dataset class for loading images.
- Data Augmentation with resizing and rotation.
- Training loop with multiple configurations (SGD/Adam, different LRs and batch sizes).
- Evaluation on validation and test sets.
- Visualization of misclassified images for error analysis.

> This part explores how a CNN trained from scratch performs with different configurations and datasets.

---

### 🧾 `dl2025_ex3_q2.py` – Part B: Fine-Tuning ResNet18
This script fine-tunes a **pretrained ResNet18** model:
- Loads ImageNet-pretrained ResNet18 and freezes feature layers.
- Replaces the final FC layer with a new classifier (Linear → ReLU → Dropout → Linear → LogSoftmax).
- Applies image transformations and normalization.
- Evaluates model performance on both regular and augmented datasets.
- Displays misclassified test images.
- Uses **ClearML** to log and compare experiments.

> This part demonstrates how transfer learning can enhance results on small datasets.

---

## 🧪 Experimental Settings

- **Loss Function**: CrossEntropyLoss
- **Optimizers**: Adam, SGD with momentum (0.9)
- **Batch Sizes**: 8, 32, 64, 128, 256
- **Learning Rates**: 0.0001, 0.001, 0.005, 0.01
- **Epochs**: 5 for CNN; 5–10 for fine-tuning
- **Augmentation**:
  - Horizontal Flip
  - Brightness/Contrast Change
  - Image Resizing (256x256 to 512x512)

---

## 🧪 Dataset

Two dataset versions were used:

| Dataset Type | Description                    | Size Estimate |
|--------------|--------------------------------|---------------|
| Regular      | Original labeled image set     | ~700 images   |
| Augmented    | Extended with transformations  | ~2100 images  |

Each dataset is structured into folders:
- Training/
- Validation/
- Testing/

---

## 📊 Results Summary

| Model Type         | Dataset       | Best Test Accuracy | Test Loss |
|--------------------|---------------|--------------------|-----------|
| CNN (from scratch) | Regular       | 80.00%             | 0.6301    |
| CNN (from scratch) | Augmented     | 73.33%             | 0.4686    |
| ResNet18 (FT)      | Regular       | 70.00%             | 0.6115    |
| ResNet18 (FT)      | Augmented     | **93.33%**         | **0.4838** |

> ✅ **Fine-tuning ResNet18 on the augmented dataset produced the highest accuracy.**

---

## 🖼️ Misclassification Analysis
- The model struggled with:
  - Low contrast or poorly lit images.
  - Objects near the mouth (e.g., cups) that resembled cigarettes.
  - Partial occlusions.
- Misclassified images were saved and displayed for each run to aid debugging.

---

## 📈 Key Insights & Lessons Learned
- Data Augmentation improved generalization significantly — especially on pretrained models.
- CNNs trained from scratch benefited from **larger batch sizes** and **SGD** in some configurations.
- **Fine-tuning only the classifier head** of ResNet18 yielded excellent results even on small datasets.
- **ClearML** enabled effective tracking and comparison of multiple training runs.
- Overfitting appeared in regular dataset when not enough data diversity was present.

---

## ⚙️ How to Run

1. Install requirements:
```bash
pip install torch torchvision matplotlib clearml
python dl_2025_ex3_q1.py
python dl2025_ex3_q2.py
```