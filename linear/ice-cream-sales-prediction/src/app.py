import streamlit as st
import torch
from model import LinearRegressionModel
from utils import load_model

# -------------------------
# Load model
# -------------------------
model = LinearRegressionModel()
model = load_model(model, "model.pth")

st.title("🍦 Ice Cream Sales Predictor")

st.markdown("### Predict profit based on temperature")

temp = st.slider("🌡 Temperature (°C)", 0.0, 50.0, 25.0)

x = torch.tensor([[temp]], dtype=torch.float32)

with torch.no_grad():
    prediction = model(x).item()

st.metric(label="💰 Expected Profit", value=f"{prediction:.2f}")