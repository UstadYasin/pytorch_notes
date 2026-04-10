# 🍦 Ice Cream Sales Prediction using PyTorch

This project predicts ice cream sales based on temperature using a simple linear regression model built with PyTorch.

---

## 📌 Problem Statement
We aim to model the relationship between temperature and ice cream sales and predict future sales based on temperature.

---

## 📊 Dataset
- Input: Temperature (°C)
- Output: Ice Cream Profits / Sales
- Source: Kaggle (Ice Cream Sales dataset)

---

## ⚙️ Tech Stack
- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib

---

## 🧠 Model
We use a simple Linear Regression model:

y = wx + b

Implemented using:
- `torch.nn.Linear(1, 1)`

---

## 🚀 Workflow

1. Load dataset
2. Normalize data
3. Convert to tensors
4. Train model using Adam optimizer
5. Evaluate performance
6. Save trained model

---

## 📉 Results

- Final Training Loss: ~6.01
- Test Loss: ~5.29

The model shows good generalization with no overfitting.

---

## 📈 Visualization

(Add your plot image here)

Example:
- Scatter plot of real vs predicted values

---

## 📦 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt