# paste this into a quick test.py and run it
from data.pets_dataset import OxfordIIITPetDataset
from torchvision import transforms

t = transforms.Compose([transforms.ToTensor()])
ds = OxfordIIITPetDataset(
    r"D:\IITM\DA6401\DA6401_Assignment2\oxford-pet",
    split="train",
    transform=t
)

print(f"Train samples: {len(ds)}")
sample = ds[0]
print(f"image shape: {sample['image'].shape}")   # expect [3, 224, 224]
print(f"label:       {sample['label']}")          # 0-36
print(f"bbox:        {sample['bbox']}")            # [cx, cy, w, h] in pixels
print(f"mask shape:  {sample['mask'].shape}")      # [224, 224]
print(f"mask unique: {sample['mask'].unique()}")   # should be subset of {0,1,2}