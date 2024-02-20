from torch import nn 
import torch 
from models.efficientsci.resdnet import ResDNetBlock
import einops

class Network_teacher(nn.Module):
    def __init__(self,in_ch=32, units=8,group_num=2,color_ch=1):
        super().__init__()
        self.color_ch = color_ch
        self.fem = nn.Sequential(
            nn.Conv3d(1, in_ch, kernel_size=(3,7,7), stride=1,padding=(1,3,3)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch*2, in_ch*4, kernel_size=3, stride=(1,2,2), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.up_conv = nn.Conv3d(in_ch*4,in_ch*8,1,1)
        self.up = nn.PixelShuffle(2)
        self.vrm = nn.Sequential(
            nn.Conv3d(in_ch*2, in_ch*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch*2, in_ch, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, color_ch, kernel_size=3, stride=1, padding=1),
        )
        self.resdnet_list = nn.ModuleList()
        for i in range(units):
            self.resdnet_list.append(ResDNetBlock(in_ch*4,group_num=group_num))

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
        # out_list = []
        if self.color_ch==3:
            x = self.bayer_init(y,Phi,Phi_s)
        else:
            meas_re = torch.div(y, Phi_s)
            meas_re = torch.unsqueeze(meas_re, 1)
            maskt = Phi.mul(meas_re)
            x = meas_re + maskt
            x = x.unsqueeze(1)
    
        out = self.fem(x)
        for resdnet in self.resdnet_list:
            out = resdnet(out)

        out = self.up_conv(out)
        out = einops.rearrange(out,"b c t h w-> b t c h w")
        out = self.up(out)
        out = einops.rearrange(out,"b t c h w-> b c t h w")
        out = self.vrm(out)

        if self.color_ch!=3:
            out = out.squeeze(1)
        # out_list.append(out)
        return out
