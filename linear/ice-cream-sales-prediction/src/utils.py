import torch

# -------------------------
# Save model
# -------------------------
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)

# Load trained model

def load_model(model, path="model.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Predict function (normalized input)
def predict(model, x):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        return model(x).numpy()
