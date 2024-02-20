import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

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

class StageBlock(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.block1 = ResNetBlock(ch)
        self.block2 = ResNetBlock(ch)

    def forward(self, x):
        out = self.block1(x)
        # out = self.block2(out)
        return out

class Unet(nn.Module):

    def __init__(self,in_ch):
        super().__init__()
        
        ch = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,ch,3,1,1),
            nn.LeakyReLU(inplace=True)
        )
        self.encoder_block1 = StageBlock(ch)      
        self.down1 = nn.Conv2d(ch,ch*2,3,2,1)
        self.encoder_block2 = StageBlock(ch*2)      
        self.down2 = nn.Conv2d(ch*2,ch*4,3,2,1)
        self.encoder_block3 = StageBlock(ch*4)      

        self.up1 = nn.Sequential(
            nn.Conv2d(ch*4,ch*8,1),
            nn.PixelShuffle(2)
        )
        self.decoder_block1 = StageBlock(ch*2)      
        self.up2 = nn.Sequential(
            nn.Conv2d(ch*2,ch*4,1),
            nn.PixelShuffle(2)
        )
        self.decoder_block2 = StageBlock(ch)      
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