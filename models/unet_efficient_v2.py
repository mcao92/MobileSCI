import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
        
class ResNetBlock(nn.Module):

    def __init__(self,width):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(width,width,3,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(width,width,3,1,1),
        )
        
    def forward(self,x):
        out = self.bottleneck(x)+x
        return out
    
class StageBlock_resnet(nn.Module):

    def __init__(self, width, depth):
        super().__init__()
        self.first_block = ResNetBlock(width)
        self.block_list = nn.ModuleList()
        for i in range(depth - 1):
            self.block_list.append(ResNetBlock(width))
    def forward(self, x):
        out = self.first_block(x)
        for res_block in self.block_list:
            out = res_block(out)
        
        return out    

class Unet(nn.Module):

    def __init__(self,in_ch,depth,width):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,width,3,1,1),
            nn.LeakyReLU(inplace=True)
        )

        self.encoder_block1 = StageBlock_resnet(width,depth)      
        self.down1 = nn.Conv2d(width,width*2,3,2,1)

        self.encoder_block2 = StageBlock_resnet(width*2,depth)      
        self.down2 = nn.Conv2d(width*2,width*4,3,2,1)

        self.encoder_block3 = StageBlock_resnet(width*4,depth)      

        self.up1 = nn.Sequential(
            nn.Conv2d(width*4,width*8,1),
            nn.PixelShuffle(2)
        )
        self.decoder_block1 = StageBlock_resnet(width*2,depth)      

        self.up2 = nn.Sequential(
            nn.Conv2d(width*2,width*4,1),
            nn.PixelShuffle(2)
        )
        self.decoder_block2 = StageBlock_resnet(width,depth)   

        self.conv2 = nn.Sequential(
            nn.Conv2d(width,in_ch,3,1,1),
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