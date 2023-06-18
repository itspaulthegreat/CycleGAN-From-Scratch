from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self,root_horse,root_zebra,transform= None):
        self.root_horse = root_horse
        self.root_zebra = root_zebra
        self.transform = transform

        self.horse_image = os.listdir(self.root_horse)
        self.zebra_image = os.listdir(self.root_zebra)

        max_len = max(len(self.zebra_image),len(self.horse_image))
        self.zebra_len = len(self.zebra_image)
        self.horse_len = len(self.horse_image)
        
    def __getitem__(self,index):
        zebra_image = self.zebra_image[index % self.zebra_len]
        horse_image = self.horse_image[index % self.horse_len]

        horse_path = os.path.join(self.root_horse + horse_image)
        zebra_path = os.path.join(self.root_zebra + zebra_image)

        horse_image = np.array(Image.open(horse_path))
        zebra_image = np.array(Image.open(zebra_path))
        
        
