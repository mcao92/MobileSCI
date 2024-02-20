from torch.utils.data import Dataset 
import numpy as np
import os
import os.path as osp
import cv2
import albumentations as A

class TrainData(Dataset):
    def __init__(self,train_data_dir,mask,in_channs=1):

        cr,mask_h,mask_w = mask.shape
        r = np.array([[1, 0], [0, 0]])
        g1 = np.array([[0, 1], [0, 0]])
        g2 = np.array([[0, 0], [1, 0]])
        b = np.array([[0, 0], [0, 1]])
        self.rgb2raw = np.zeros([3, mask_h, mask_w])
        self.rgb2raw[0, :, :] = np.tile(r, (mask_h // 2, mask_w // 2))
        self.rgb2raw[1, :, :] = np.tile(g1, (mask_h // 2, mask_w // 2)) + np.tile(g2, (
            mask_h // 2, mask_w // 2))
        self.rgb2raw[2, :, :] = np.tile(b, (mask_h // 2, mask_w // 2))
        self.color_flag=False
        if in_channs==3:
            self.color_flag=True
            
        self.data_dir= train_data_dir
        self.data_list = os.listdir(train_data_dir)
        self.img_files = []
        
        self.mask = mask
        self.ratio,self.resize_w,self.resize_h = mask.shape
        for image_dir in os.listdir(train_data_dir):
            train_data_path = osp.join(train_data_dir,image_dir)
            data_path = os.listdir(train_data_path)
            data_path.sort()
            for sub_index in range(len(data_path)-self.ratio):
                sub_data_path = data_path[sub_index:]
                meas_list = []
                count = 0
                for image_name in sub_data_path:
                    meas_list.append(osp.join(train_data_path,image_name))
                    if (count+1)%self.ratio==0:
                        self.img_files.append(meas_list)
                        meas_list = []
                    count += 1
        
    def __getitem__(self,index):
        gt = np.zeros([self.ratio, self.resize_h, self.resize_w],dtype=np.float32)
        if self.color_flag is True:
            gt = np.zeros([3,self.ratio,self.resize_h, self.resize_w],dtype=np.float32)
        meas = np.zeros([self.resize_h, self.resize_w],dtype=np.float32)
        gt_images_list = []
        p = np.random.randint(0,10)>5
        image = cv2.imread(self.img_files[index][0])
        image_h,image_w = image.shape[:2]
        mask_h,mask_w= self.mask.shape[1:]
     
        crop_h = np.random.randint(mask_h//2,image_h)
        crop_w = np.random.randint(mask_w//2,image_w)
        crop_p = np.random.randint(0,10)>5
        flip_p = np.random.randint(0,10)>5
        # rot_p = np.random.randint(0,10)>5
        transform = A.Compose([
            A.CenterCrop(height=crop_h,width=crop_w,p=crop_p),
            A.HorizontalFlip(p=flip_p),
            # A.RandomRotate90(p=rot_p),
            A.Resize(self.resize_h,self.resize_w)
        ])
        rotate_flag = np.random.randint(0,10)>5
        for i,image_path in enumerate(self.img_files[index]):
            image = cv2.imread(image_path)
            
            # im_h,im_w = image.shape[:2]
            if image_h<crop_h or image_w<image_w:
                image = cv2.resize(image,(crop_w,crop_h))
            transformed = transform(image=image)
            image = transformed["image"]
            # print(rotate_flag)
            if rotate_flag:
                image = cv2.flip(image, 1)
                image = cv2.transpose(image)

            # cv2.imwrite("tt/"+str(i)+"src_image.png",image)
            # cv2.waitKey(0)
            if not self.color_flag:
                pic_t = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)[:,:,0]
                pic_t = pic_t.astype(np.float32)
                pic_t /= 255.
            else:
                pic_t = np.sum(image[:,:,::-1]*np.transpose(self.rgb2raw,(1,2,0)),axis=2)
                pic_t /= 255.
                temp = image[:,:,::-1]
                temp = temp.astype(np.float32)
                # temp = (temp-self.mean)/self.std
                temp = temp/255.
                temp = temp.transpose(2,0,1)
            pic_t = pic_t.astype(np.float32)
            # gt_images_list.append(pic_t)
            mask_t = self.mask[i, :, :]
            if self.color_flag:
                gt[:,i] = temp 
            else:
                gt[i] = pic_t 
            meas += np.multiply(mask_t.numpy(), pic_t)
        
        return gt,meas
    def __len__(self,):
        return len(self.img_files)