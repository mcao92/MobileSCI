from torch import nn 
import torch 
import einops
from models.unet_efficient_v2 import Unet 

class Network(nn.Module):
    def __init__(self,color_ch=1,depth=2,width=64):
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
