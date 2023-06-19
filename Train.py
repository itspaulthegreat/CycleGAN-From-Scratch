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
from tqdm import tqdm
from torchvision.utils import save_image
import sys 
from utils import save_checkpoint, load_checkpoint


def train(gen_h,gen_z,disc_h,disc_z,opti_gen,opti_disc,g_scaler,d_scaler,MSELoss,L1Loss,train_loader,val_loader):
    print(config.DEVICE)
    H_reals = 0
    H_fakes = 0
    loop = tqdm(train_loader, leave=True)

    for idx, (zebra,horse) in enumerate(loop):
        horse = horse.to(config.DEVICE)
        zebra = zebra.to(config.DEVICE)

        with torch.cuda.amp.autocast():

            fake_horse = gen_h(zebra)
            disc_Hreal = disc_h(horse)
            disc_Hfake = disc_h(fake_horse.detach())
            H_reals += disc_Hreal.mean().item()
            H_fakes += disc_Hfake.mean().item()
            D_Hrealloss = MSELoss(disc_Hreal,torch.ones_like(disc_Hreal))
            D_Hfakeloss = MSELoss(disc_Hfake,torch.zeros_like(disc_Hfake))
            D_HLoss = D_Hrealloss + D_Hfakeloss

            fake_zebra = gen_z(horse)
            disc_Zreal = disc_z(zebra)
            disc_Zfake = disc_z(fake_zebra.detach())
            D_Zrealloss = MSELoss(disc_Zreal,torch.ones_like(disc_Zreal))
            D_Zfakeloss = MSELoss(disc_Zfake,torch.zeros_like(disc_Zfake))
            D_ZLoss = D_Zrealloss + D_Zfakeloss

            D_Loss = (D_HLoss + D_ZLoss) / 2
        
        opti_disc.zero_grad()
        d_scaler.scale(D_Loss).backward()
        d_scaler.step(opti_disc)
        d_scaler.update()


        with torch.cuda.amp.autocast():

            disc_Zfake = disc_z(fake_horse)
            disc_Hfake = disc_h(fake_zebra)
            gen_zloss = MSELoss(disc_Zfake,torch.ones_like(disc_Zfake))
            gen_hloss = MSELoss(disc_Hfake,torch.ones_like(disc_Hfake))
            
            cyclezebra = gen_z(fake_horse)
            cyclehorse = gen_h(fake_zebra)
            cyle_ZLoss = L1Loss(zebra,cyclezebra)
            cyle_HLoss = L1Loss(horse,cyclehorse)

            # identityzebra = gen_z(zebra)
            # identityhorse = gen_h(horse)
            # identity_ZLoss = L1Loss(zebra,identityzebra)
            # identity_HLoss = L1Loss(horse,identityhorse)



        G_Loss = (
            gen_zloss + gen_hloss + (cyle_ZLoss * config.LAMBDA_CYCLE) + (cyle_HLoss * config.LAMBDA_CYCLE) #+ (identity_ZLoss * config.LAMBDA_IDENTITY) + (identity_HLoss * config.LAMBDA_IDENTITY)
        )

        opti_gen.zero_grad()
        g_scaler.scale(G_Loss).backward()
        g_scaler.step(opti_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


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
    
    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_h,
            opti_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_z,
            opti_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_h,
            opti_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_z,
            opti_disc,
            config.LEARNING_RATE,
        )

    train_dataset = HorseZebraDataset(
        root_horse = "data/train/horse/",
        root_zebra =  "data/train/zebra/",
        transform= config.TRANSFORM
    )

    val_dataset = HorseZebraDataset(
        root_horse="data/val/horse/",
        root_zebra =  "data/val/zebra/",
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
    
    for epoch in range(config.NUM_EPOCHS):

        train(gen_h,gen_z,disc_h,disc_z,opti_gen,opti_disc,g_scaler,d_scaler,MSELoss,L1Loss,train_loader,val_loader)

        if config.SAVE_MODEL:
            save_checkpoint(gen_h, opti_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_z, opti_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_h, opti_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_z, opti_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":

    main()