# debug_localizer.py
import torch
from torchvision import transforms
from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = VGG11Localizer().to(device)
model.load_state_dict(torch.load("checkpoints/localizer.pth", map_location=device))
model.eval()

# Load a few val samples
t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ds = OxfordIIITPetDataset(r"./oxford-pet", split="val", transform=t)

print("Sample predictions vs ground truth (normalized [0,1]):")
with torch.no_grad():
    for i in range(5):
        img = ds[i]["image"].unsqueeze(0).to(device)
        gt  = ds[i]["bbox"]
        pred = model(img)[0].cpu()
        print(f"  pred: {pred.numpy().round(3)}  gt: {gt.numpy().round(3)}")