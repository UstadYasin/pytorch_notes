import torch
import pandas as pd
from model import LinearRegressionModel
from utils import save_model

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("ice_cream.csv")

X = df["Temperature"].values
y = df["Ice Cream Profits"].values

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# -------------------------
# Model
# -------------------------
model = LinearRegressionModel()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# -------------------------
# Training loop
# -------------------------
epochs = 5000

for epoch in range(epochs):
    model.train()

    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------------------------
# Save model
# -------------------------
save_model(model, "model.pth")