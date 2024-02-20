import os 
import os.path as osp 
from torch.utils.data import Dataset
import h5py 
import numpy as np 
import torch 
import cv2 
import scipy.io as scio 
import einops 

class TestData(Dataset):
    def __init__(self,data_path,mask):
        self.data_path = data_path
        self.data_list = os.listdir(data_path)
        self.mask = mask
        compression_ratio,mask_h,mask_w = mask.shape
        r = np.array([[1, 0], [0, 0]])
        g1 = np.array([[0, 1], [0, 0]])
        g2 = np.array([[0, 0], [1, 0]])
        b = np.array([[0, 0], [0, 1]])
        self.rgb2raw = np.zeros([3, mask_h, mask_w])
        self.rgb2raw[0, :, :] = np.tile(r, (mask_h // 2, mask_w // 2))
        self.rgb2raw[1, :, :] = np.tile(g1, (mask_h // 2, mask_w // 2)) + np.tile(g2, (
            mask_h // 2, mask_w // 2))
        self.rgb2raw[2, :, :] = np.tile(b, (mask_h // 2, mask_w // 2))

    def __getitem__(self,index):
        pic = scio.loadmat(osp.join(self.data_path,self.data_list[index]))
        pic = pic["orig"]
        pic = einops.rearrange(pic,"h w c b->b c h w")
        pic_gt = np.zeros([pic.shape[0] // 8,3,8,1080,1920])
        bayer_gt = np.zeros([pic.shape[0] // 8,8,1080,1920])
        for jj in range(pic.shape[0]):
            if jj % 8 == 0:
                meas_t = np.zeros([1080,1920])
                n = 0
            pic_t = pic[jj]

            # pic_t = np.rot90(pic_t,axes=(1,2))
            # pic_t = np.flip(pic_t,axis=1)
            pic_t= pic_t.astype(np.float32)
            
            pic_t /= 255.
            temp = pic_t
            # temp = temp.transpose(1,2,0)
            # temp = (temp-self.mean)/self.std
            # temp = temp.transpose(2,0,1)
            pic_t = np.sum(pic_t*self.rgb2raw,axis=0)
            
        
            mask_t = self.mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // 8,:,n] = temp
            bayer_gt[jj//8,n] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % 8 == 0 and jj != 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        return torch.from_numpy(meas),pic_gt,bayer_gt
    def __len__(self,):
        return len(self.data_list)