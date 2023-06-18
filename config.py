import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_CHANNELS = 3
TRAIN_DIR = "data/train"

transform = A.Compose([
    A.Resize(height=256,width=256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5 for _ in range(IMG_CHANNELS)],std=[0.5 for _ in range(IMG_CHANNELS)]),
    ToTensorV2()
])