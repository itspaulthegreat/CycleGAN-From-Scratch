import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,act = True,down = True, **kwargs):
        super.__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,padding_mode="reflect",**kwargs) if down else nn.ConvTranspose2d(in_channels,out_channels,**kwargs)  ,
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

    def forward(self,x):
        return self.layer(x)

class ResBlock(nn.Module):
    def __init__(self,in_channels):
        super.__init__()

        self.resblock = nn.Sequential(
            Block(in_channels,in_channels,kernel_size = 3 , padding =1),
            Block(in_channels,in_channels,act=False,kernel_size = 3 , padding =1)
        )

    def forward(self,x):
        return x + self.resblock(x)
    

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
