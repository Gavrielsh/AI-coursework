# 📉 Optimization Algorithms – Gradient Descent Variants

## 🧠 Overview

This project explores several gradient-based optimization methods and analyzes their convergence behavior on different types of functions. The goal is to visualize and compare how different optimizers perform on convex, non-convex, and oscillatory functions.

> ✅ Implemented as part of a Deep Learning university course by Gavriel Shalem.

---

## 🧪 Objective

Evaluate and compare the following optimization methods:
- **Stochastic Gradient Descent (SGD)**
- **Momentum**
- **Nesterov Accelerated Gradient (NAG)**
- **AdaGrad**

Each method is tested on three types of functions:
1. `x²` – Simple convex function with a clear minimum.
2. `x⁴` – Convex function with small gradients near the minimum.
3. `sin(x)` – Oscillatory function with many local minima.

Additionally, extreme values (large and small) are tested to evaluate stability.

---

## 🧩 Key AI & Optimization Concepts Used

- **Gradient Descent & Variants**
- **Momentum & Nesterov Update Rules**
- **Learning Rate Scheduling via AdaGrad**
- **Convergence Visualization**
- **Edge Case Robustness Testing**
- **Mathematical Function Derivatives**

---

## 🔍 Project Structure

### 🧾 `ex2.py`
Contains:
- Definitions of the mathematical functions and their derivatives.
- Implementations of the four optimizers.
- Training and convergence loop.
- Visualization using `matplotlib` for comparing convergence.
- Final value analysis after 200 steps.
- Edge case testing with extreme starting values (`1e6`, `1e-6`, etc.).

### 📄 `ex2.pdf`
- Formal report (in Hebrew) summarizing:
  - Experiment setup
  - Graph analysis
  - Comparative performance between optimizers
  - Strengths and weaknesses of each method
  - Conclusions and edge case interpretation

---

## 🧪 Experimental Setup

- **Initial Point**: `x = 10.0`
- **Learning Rate**: `0.01`
- **Steps**: `200` iterations per method
- **Momentum Coefficient**: `0.9` for Momentum/NAG
- **AdaGrad**: Uses learning rate adjustment with history tracking

---

## 📈 Results & Graphs
![alt text](image.png)
### 🔵 `x²` – Simple Convex
- **Best**: NAG and Momentum converge quickly.
- **Worst**: AdaGrad slows too quickly due to LR decay.
- **SGD**: Slow but stable.

### 🔵 `x⁴` – Flat Gradients Near Minimum
- **SGD**: Struggles with slow progress.
- **Momentum**: Strong overshooting.
- **NAG**: Smoother convergence.
- **AdaGrad**: Very slow convergence.

### 🔵 `sin(x)` – Many Local Minima
- **NAG** & **Momentum**: Navigate local minima better.
- **SGD**: Gets trapped.
- **AdaGrad**: Learning rate drops too fast to escape minima.

### 🧪 Edge Case Tests:
- Robustness tested on:
  - `x = 1e6`, `-1e6`, `1e-6`, `-1e-6`
- Most stable: **NAG** and **SGD** in extreme regimes.

---

## 📌 Key Takeaways

| Method   | Strengths                                   | Weaknesses                                  |
|----------|---------------------------------------------|----------------------------------------------|
| **SGD**      | Simple, reliable on smooth convex functions | Slow on flat gradients, poor in non-convex domains |
| **Momentum** | Faster convergence than SGD               | May overshoot, requires careful tuning       |
| **NAG**      | Anticipates future gradients, fast        | More computationally expensive               |
| **AdaGrad**  | Good for sparse gradients, auto-LR tuning | Too aggressive LR decay                      |

---

## 🖼️ Sample Graphs
The script will generate and display 3 plots:
- `x²` convergence
- `x⁴` convergence
- `sin(x)` convergence

All methods are color-coded and plotted across 200 iterations.

---

## 🛠️ How to Run

1. Install Python requirements:
```bash
pip install numpy matplotlib
