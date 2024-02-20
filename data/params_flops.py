from thop import profile
from models.network import Network
import torch 
import time
import os 
from torch.cuda.amp import autocast, GradScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda")

b,f,h,w = 1,8,256,256
# b,f,h,w = 1,50,512,512
# b,f,h,w = 1,28,1080,1920
model = Network().to(device)

Phi = torch.randn(b, f, h, w).to(device)
Phi_s = torch.randn(b, h, w).to(device)
meas = torch.randn(b, h, w).to(device)
# model.load_state_dict(torch.load("weights/student_1.pth"))
# scaler = GradScaler()
# with autocast():
with torch.no_grad():
    out = model(meas,Phi,Phi_s)
macs, params = profile(model, inputs=(meas,Phi,Phi_s))
with torch.no_grad():
    for i in range(10):
        start = time.time()
        out = model(meas,Phi,Phi_s)
        torch.cuda.synchronize()
        end = time.time()
        print("forward time: ",end-start)
print("para: {:.6f} M, FLOATs: {:.6f} G. ".format(params/1e6,macs/1e9))
