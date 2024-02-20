import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

from models.efficientformer import FFN,AttnFFN

class GhostModule(nn.Module):
    def __init__(self, inp, kernel_size=3, dw_size=3, stride=1):
        super(GhostModule, self).__init__()
        init_channels = inp//2
        new_channels = init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.LeakyReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out

class StageBlock_ghost(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.block1 = GhostModule(ch)
        self.block2 = GhostModule(ch)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out
        
class ResNetBlock(nn.Module):

    def __init__(self,ch):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch,ch,3,1,1),
        )
        
    def forward(self,x):
        out = self.bottleneck(x)+x
        return out
    
class StageBlock_resnet(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.block1 = ResNetBlock(ch)
        self.block2 = ResNetBlock(ch)
        self.block3 = ResNetBlock(ch)
        self.block4 = ResNetBlock(ch)
        self.block5 = ResNetBlock(ch)
        self.block6 = ResNetBlock(ch)
        # self.block7 = ResNetBlock(ch)
        # self.block8 = ResNetBlock(ch)
        # self.block9 = ResNetBlock(ch)
        # self.block10 = ResNetBlock(ch)
        # self.block11 = ResNetBlock(ch)
        # self.block12 = ResNetBlock(ch)
        # self.block13 = ResNetBlock(ch)
        # self.block14 = ResNetBlock(ch)
        # self.block15 = ResNetBlock(ch)
        # self.block16 = ResNetBlock(ch)
        # self.block17 = ResNetBlock(ch)
        # self.block18 = ResNetBlock(ch)
       
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        # out = self.block7(out)
        # out = self.block8(out)
        # out = self.block9(out)
        # out = self.block10(out)
        # out = self.block11(out)
        # out = self.block12(out)
        # out = self.block13(out)
        # out = self.block14(out)
        # out = self.block15(out)
        # out = self.block16(out)
        # out = self.block17(out)
        # out = self.block18(out)
        
        return out    

class StageBlock_ffn(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.block1 = FFN(dim=ch, mlp_ratio=4.,norm=False)
        self.block2 = FFN(dim=ch, mlp_ratio=4.,norm=False)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out

class StageBlock_attnffn(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.block1 = AttnFFN(dim=ch, mlp_ratio=4.,norm=True,stride=2)
        self.block2 = AttnFFN(dim=ch, mlp_ratio=4.,norm=True,stride=2)
        # self.block3 = GhostModule(ch)
        # self.block4 = GhostModule(ch)

    def forward(self, x):
        out = self.block1(x)
        # out = self.block3(out)
        out = self.block2(out)
        # out = self.block4(out)
        return out

class Unet(nn.Module):

    def __init__(self,in_ch):
        super().__init__()
        
        ch = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,ch,3,1,1),
            nn.LeakyReLU(inplace=True)
        )

        self.encoder_block1 = StageBlock_resnet(ch)      
        self.down1 = nn.Conv2d(ch,ch*2,3,2,1)

        self.encoder_block2 = StageBlock_resnet(ch*2)      
        self.down2 = nn.Conv2d(ch*2,ch*4,3,2,1)

        self.encoder_block3 = StageBlock_resnet(ch*4)      

        self.up1 = nn.Sequential(
            nn.Conv2d(ch*4,ch*8,1),
            nn.PixelShuffle(2)
        )
        self.decoder_block1 = StageBlock_resnet(ch*2)      

        self.up2 = nn.Sequential(
            nn.Conv2d(ch*2,ch*4,1),
            nn.PixelShuffle(2)
        )
        self.decoder_block2 = StageBlock_resnet(ch)   

        self.conv2 = nn.Sequential(
            nn.Conv2d(ch,in_ch,3,1,1),
        )
        
    def forward(self, x):
        
        conv1_out = self.conv1(x)
        en_b1 = self.encoder_block1(conv1_out)
        d1 = self.down1(en_b1)
        en_b2 = self.encoder_block2(d1)
        d2 = self.down2(en_b2)
        en_b3 = self.encoder_block3(d2)
        u1 = self.up1(en_b3)
        de_b1 = self.decoder_block1(u1+en_b2)
        u2 = self.up2(de_b1)
        de_b2 = self.decoder_block2(u2+en_b1)
        out = self.conv2(de_b2+conv1_out)
        return out