import torch
import torch.nn
import src
from src.Discriminator import Discriminator
from src.Generator import Generator
import config
import torch.optim as optim
from dataset import HorseZebraDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


def train(gen_h,gen_z,disc_h,disc_z,opti_gen,opti_disc,g_scaler,d_scaler,MSELoss,L1Loss,train_loader,val_loader):
    
    H_reals = 0
    H_fakes = 0
    loop = tqdm(train_loader, leave=True)

    for idx, (zebra,horse) in enumerate(loop):
        horse.to(config.DEVICE)
        zebra.to(config.DEVICE)

        with torch.cuda.amp.autocast():

            fake_horse = gen_h(zebra)
            disc_Hreal = disc_h(horse)
            disc_Hfake = disc_h(fake_horse)
            


def main():
    gen_h = Generator(in_channels = config.IMG_CHANNELS,residual_blocks=config.RESIDUAL_BLOCKS).to(device=config.DEVICE)
    disc_h= Discriminator(in_channels= config.IMG_CHANNELS).to(device=config.DEVICE)
    gen_z = Generator(in_channels = config.IMG_CHANNELS,residual_blocks=config.RESIDUAL_BLOCKS).to(device=config.DEVICE)
    disc_z= Discriminator(in_channels= config.IMG_CHANNELS).to(device=config.DEVICE)

    
    opti_gen = optim.Adam(
        list(gen_h.parameters()) + list(gen_z.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5,0.999)
    )

    opti_disc = optim.Adam(
        list(disc_h.parameters()) + list(disc_z.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5,0.999)
    )
    
    MSELoss = nn.BCELoss()
    L1Loss = nn.L1Loss()

    train_dataset = HorseZebraDataset(
        root_horse = "data/train/horse",
        root_zebra =  "data/train/zebra",
        transform= config.TRANSFORM
    )

    val_dataset = HorseZebraDataset(
        root_horse="data/val/horse",
        root_zebra =  "data/val/zebra",
        transform= config.TRANSFORM
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    for epoch in config.NUM_EPOCHS:

        train(gen_h,gen_z,disc_h,disc_z,opti_gen,opti_disc,g_scaler,d_scaler,MSELoss,L1Loss,train_loader,val_loader)

if __name__ == "__main__":

    main()