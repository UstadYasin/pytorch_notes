import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from model import LinearRegressionModel
from utils import load_model

# -------------------------
# 1. Load data
# -------------------------
df = pd.read_csv("ice_cream.csv")

X = df["Temperature"].values
y = df["Ice Cream Profits"].values

# -------------------------
# 2. Load model
# -------------------------
model = LinearRegressionModel()
model = load_model(model, "model.pth")

# -------------------------
# 3. Convert to tensor (NO normalization)
# -------------------------
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

# -------------------------
# 4. Predict
# -------------------------
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

# -------------------------
# 5. Plot using seaborn
# -------------------------
sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))

sns.scatterplot(x=X, y=y, label="Actual Data")
sns.lineplot(x=X, y=y_pred.flatten(), color="red", label="Model Prediction")

plt.title("Ice Cream Sales vs Temperature (No Normalization)")
plt.xlabel("Temperature")
plt.ylabel("Ice Cream Profits")

plt.legend()
plt.show()