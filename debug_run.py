import torch
from models.multitask import MultiTaskPerceptionModel

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = MultiTaskPerceptionModel().to(device)

# Dummy input
x = torch.randn(1, 3, 224, 224).to(device)

# Run forward
output = model(x)

print("\nFINAL OUTPUT:")
print(output["localization"])