from posixpath import basename
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.functional import img_to_tensor
import json
import os
from glob import glob
import random
import numpy as np
import cv2
try:    
    from transforms.transform import create_transform_resize
except Exception:
    from .transforms.transform import create_transform_resize
class FaceForensicsDataset(Dataset):
    def __init__(self,is_real,frame_dir,num_frames,split_dir,split,bbox_name,transform,size=256):
        super().__init__()
        self.frame_dir = frame_dir
        self.num_frames = num_frames
        self.split = split
        self.transform = transform
        self.split_name = self.plain_json(split_dir) # 划分数据集
        self.frame_name = frame_dir.split('/')[-2]
        self.bbox_name = bbox_name
        self.size=size
        self.data = []
        fake_methods = ['FaceShifter', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        real_methods = ['original_sequences/youtube']

        if self.split == 'train':
            if is_real:
                self.generate_data(real_methods)
            else:
                self.generate_data(fake_methods)
        else:
            self.generate_data(fake_methods+real_methods) # 验证集真假图片都有
        self.normalize = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
       
    
    def generate_data(self,methods):
        for method in methods:
            frame_dir = self.frame_dir.replace('method',method)
            for _,dirs,__ in os.walk(frame_dir):
                for dir in dirs:
                    if dir not in self.split_name and dir.split('_')[0] not in self.split_name:
                        continue
                    video_path = frame_dir+dir
                    tmp = glob(video_path+'/*.png')
                    if 'original' in method:
                        each_num_frame =self.num_frames
                    else:
                        each_num_frame =self.num_frames//5
                    tmp = sorted(random.sample(tmp,min(each_num_frame,len(tmp))))
                    for p in tmp:
                        if 'original' in p:
                            label = 0
                        else:
                            label = 1
                        self.data.append((p,label)) # 保存frame_path和label
                break
    
    def plain_json(self,split_dir):
        f_train = open(split_dir+'train.json',"r")
        f_val = open(split_dir+'val.json',"r")
        f_test = open(split_dir+'test.json',"r")
        train_json = json.load(f_train)
        val_json = json.load(f_val)
        test_json =json.load(f_test)
        train_name = []
        val_name = []
        test_name = []
        for x,y in train_json:
            train_name.append(x)
            train_name.append(y)
        for x,y in val_json:
            val_name.append(x)
            val_name.append(y)
        for x,y in test_json:
            test_name.append(x)
            test_name.append(y)
        if self.split =='train':
            return train_name
        elif self.split=='val':
            return val_name
        elif self.split=='test':
            return test_name

    def load_train_sample(self, img_path):
        try:  
            basename = os.path.basename(img_path)
            bbox_path = img_path.replace('/'+basename,'.npy').replace(self.frame_name,self.bbox_name) # !!!!!bbox_path is similar to img_path
            bbox_dict = np.load(bbox_path,allow_pickle=True).item()
            img_idx = int(basename.split('.')[0])
            x0,x1,y0,y1 = bbox_dict[img_idx]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = img[x0:x1,y0:y1][:]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB chann
            if self.transform:
                data = self.transform(image=img)
                img = data['image']
            img = img_to_tensor(img, self.normalize)
            return img
        except Exception as e:
            print(img_path, ' error!', e)
            return torch.randn((3, self.size, self.size))


    def load_val_sample(self, img_path):
        try:
            basename = os.path.basename(img_path)
            bbox_path = img_path.replace('/'+basename,'.npy').replace(self.frame_name,self.bbox_name) # !!!!!bbox_path is similar to img_path
            bbox_dict = np.load(bbox_path,allow_pickle=True).item()
            img_idx = int(basename.split('.')[0])
            x0,x1,y0,y1 = bbox_dict[img_idx]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = img[x0:x1,y0:y1][:]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB chann
            if self.transform:
                data = self.transform(image=img)
                img = data['image']
            img = img_to_tensor(img, self.normalize)
            return img
        except Exception as e:
            print(img_path, ' error!', e)
            return torch.randn((3, self.size, self.size))

    def __getitem__(self, index):
        path,lab = self.data[index]
        if self.split == 'train':
            img = self.load_train_sample(path)
            return lab, img, path
        else:
            img = self.load_val_sample(path)
            return lab, img, path

    def __len__(self):
        return len(self.data)
    
    def collate_train_function(self,data):
        transposed_data = list(zip(*data))
        lab, img,img_path = transposed_data[0], transposed_data[1], transposed_data[2]
        img = torch.stack(img, 0)
        lab = torch.from_numpy(np.stack(lab, 0))
        return lab, img, img_path

    def collate_val_function(self,data):
        transposed_data = list(zip(*data))
        lab,img, img_path = transposed_data[0], transposed_data[1], transposed_data[2]
        img = torch.stack(img, 0)
        lab = torch.from_numpy(np.stack(lab, 0))
        return lab, img, img_path

if __name__ == '__main__':
    split_dir = '/raid/lpy/data/FaceForensics++/splits/'
    frame_dir = '/raid/lpy/data/FaceForensics++/method/c23/frames_64/'
    bbox_name = 'bbox_64_1.5'

   #  f= open('test1.txt',"w+")
    dataset = FaceForensicsDataset(True,frame_dir,1, split_dir,'train',bbox_name,create_transform_resize(256))
    lab,img,img_path = dataset[0]
    print(img_path)