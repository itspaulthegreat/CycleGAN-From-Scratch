import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,act = True,down = True, **kwargs):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,padding_mode="reflect",**kwargs) if down else nn.ConvTranspose2d(in_channels,out_channels,**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

    def forward(self,x):
        return self.layer(x)

class ResBlock(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.resblock = nn.Sequential(
            Block(in_channels,in_channels,kernel_size = 3 , padding =1),
            Block(in_channels,in_channels,act=False,kernel_size = 3 , padding =1)
        )

    def forward(self,x):
        return x + self.resblock(x)
    

class Generator(nn.Module):
    def __init__(self,in_channels, features = 64,residual_blocks=9):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,features,kernel_size = 7,stride=1,padding=3,padding_mode="reflect"),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        )

        
        self.down = nn.ModuleList(
            [
                Block(features,features*2,kernel_size=3,stride=2,padding=1),
                Block(features*2,features*4,kernel_size=3,stride=2,padding=1)
            ]

        )

        self.resnet = nn.Sequential(*[ResBlock(features*4) for _ in range(residual_blocks)])   # using list in nn.sequential we have to use nn.sequential(*list) to unwrap the list


        self.up = nn.ModuleList(
            [
                Block(features*4,features*2,down = False,kernel_size =3,stride=2,padding=1,output_padding=1),
                Block(features*2,features*1,down = False,kernel_size =3,stride=2,padding=1,output_padding=1)
            ]
        )

        self.last = nn.Conv2d(features*1,in_channels,kernel_size = 7,stride=1,padding=3,padding_mode="reflect")

    def forward(self,x):
        x = self.initial(x)
        for block in self.down:
            x = block(x)

        x = self.resnet(x)

        for block in self.up:
            x = block(x)
        
        x = self.last(x)

        return torch.tanh(x)
    

def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)


if __name__ == "__main__":
    test()