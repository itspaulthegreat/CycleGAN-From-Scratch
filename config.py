import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_CHANNELS = 3
TRAIN_DIR = "data/train"
RESIDUAL_BLOCKS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
NUM_WORKERS = 5
NUM_EPOCHS = 30
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0
TRANSFORM = A.Compose(
    [
    A.Resize(height=256,width=256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5 for _ in range(IMG_CHANNELS)],std=[0.5 for _ in range(IMG_CHANNELS)]),
    ToTensorV2()
],
additional_targets={"image0":"image"}
)