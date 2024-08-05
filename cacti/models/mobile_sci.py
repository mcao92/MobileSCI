from torch import nn 
import torch 
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from .builder import MODELS
from einops import rearrange

class SplitPointMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        hidden_dim = int(dim//2 * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(dim//2, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim//2, 1, 1, 0),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.fc(x1)
        x = torch.cat([x1, x2], dim=1)
        return rearrange(x, 'b (g d) h w -> b (d g) h w', g=8)
    
class SMLayer(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super().__init__()
        self.spatial = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)
        self.mlp1 = SplitPointMlp(dim, mlp_ratio)

    def forward(self, x):
        x = self.mlp1(x) + x
        x = self.spatial(x)
        return x

class FMBlock(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super().__init__()
        self.net = nn.Sequential(
            SMLayer(dim, kernel_size, mlp_ratio),
            SMLayer(dim, kernel_size, mlp_ratio),
        )

    def forward(self, x):
        x = self.net(x) + x
        return x

class StageBlock_FMBlock(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.first_block = FMBlock(width, kernel_size=3)
        self.block_list = nn.ModuleList()
        for i in range(depth - 1):
            self.block_list.append(FMBlock(width, kernel_size=3))
    def forward(self, x):
        out = self.first_block(x)
        for res_block in self.block_list:
            out = res_block(out)
        return out  
    
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

        self.encoder_block3 = StageBlock_FMBlock(width*4,1)  

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

@MODELS.register_module 
class mobile_sci(nn.Module):
    def __init__(self,color_ch=1,depth=6,width=64):
        super().__init__()
        self.color_ch = color_ch
        self.unet = Unet(in_ch=8,depth=depth,width=width)

    def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,ba[0]::2,ba[1]::2]
        y_bayer = einops.rearrange(y_bayer,"b h w ba->(b ba) h w")
        Phi_bayer = einops.rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = einops.rearrange(Phi_s_bayer,"b h w ba->(b ba) h w")

        meas_re = torch.div(y_bayer, Phi_s_bayer)
        meas_re = torch.unsqueeze(meas_re, 1)
        maskt = Phi_bayer.mul(meas_re)
        x = meas_re + maskt
        x = einops.rearrange(x,"(b ba) f h w->b f h w ba",b=b)

        x_bayer = torch.zeros(b,f,h,w).to(y.device)
        for ib in range(len(bayer)): 
            ba = bayer[ib]
            x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
        x = x_bayer.unsqueeze(1)
        return x
    def forward(self, y,Phi,Phi_s):
        out_list = []
        if self.color_ch==3:
            x = self.bayer_init(y,Phi,Phi_s)
        else:
            meas_re = torch.div(y, Phi_s)
            meas_re = torch.unsqueeze(meas_re, 1)
            maskt = Phi.mul(meas_re)
            x = meas_re + maskt
            # x = x.unsqueeze(1)
    
        out = self.unet(x)
        return out