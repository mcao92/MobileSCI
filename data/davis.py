from torch.utils.data import Dataset
import os
import os.path as osp
import torch
import scipy.io as scio
import numpy as np 

class TrainData(Dataset):
    def __init__(self, path):
        self.data = []
        if os.path.exists(path):
            dir_list = os.listdir(path)
            groung_truth_path = path + '/gt'
            measurement_path = path + '/measurement'

            if os.path.exists(groung_truth_path) and os.path.exists(measurement_path):
                groung_truth = os.listdir(groung_truth_path)
                measurement = os.listdir(measurement_path)
                self.data = [{'groung_truth': groung_truth_path + '/' + groung_truth[i],
                              'measurement': measurement_path + '/' + measurement[i]} for i in range(len(groung_truth))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

        groung_truth, measurement = self.data[index]["groung_truth"], self.data[index]["measurement"]

        gt = scio.loadmat(groung_truth)
        meas = scio.loadmat(measurement)
        if "patch_save" in gt:
            gt = torch.from_numpy(gt['patch_save'] / 255)
        elif "p1" in gt:
            gt = torch.from_numpy(gt['p1'] / 255)
        elif "p2" in gt:
            gt = torch.from_numpy(gt['p2'] / 255)
        elif "p3" in gt:
            gt = torch.from_numpy(gt['p3'] / 255)

        meas = torch.from_numpy(meas['meas'] / 255)

        gt = gt.permute(2, 0, 1)
        return gt, meas

    def __len__(self):

        return len(self.data)

class TestData(Dataset):
    def __init__(self,data_path,mask):
        self.data_path = data_path
        self.data_list = os.listdir(data_path)
        self.mask = mask
        self.cr = self.mask.shape[0]

    def __getitem__(self,index):
        pic = scio.loadmat(osp.join(self.data_path,self.data_list[index]))
        if "orig" in pic:
            pic = pic['orig']
        elif "patch_save" in pic:
            pic = pic['patch_save']
        elif "p1" in pic:
            pic = pic['p1']
        elif "p2" in pic:
            pic = pic['p2']
        elif "p3" in pic:
            pic = pic['p3']
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // self.cr, self.cr, 256, 256])
        for jj in range(pic.shape[2]):
            if jj // self.cr>=pic_gt.shape[0]:
                break
            if jj % self.cr == 0:
                meas_t = np.zeros([256, 256])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = self.mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // self.cr, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == (self.cr-1):
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % self.cr == 0 and jj != self.cr:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        return torch.from_numpy(meas),pic_gt
    def __len__(self,):
        return len(self.data_list)