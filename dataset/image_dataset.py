from email.mime import image
import torch
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
import cv2
from tqdm import tqdm
import os
import numpy as np
try:    
    from transforms.transform import create_transform_resize
except Exception:
    from .transforms.transform import create_transform_resize

def read_annotations(data_path):
    # lines = map(str.strip, open(data_path).readlines())
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            try:
                line = line.strip()
                sample_path = line[:-2]
                if not os.path.exists(sample_path):
                    print(sample_path)
                    continue
                label = line[-1]
                label = int(label)
                tmp = []
                tmp.append(sample_path)
                tmp.append(label)
                data.append(tmp)
            except:
                print(line, "fail!")
                continue
        print("{} num: {}".format(data_path, len(data)))
    return data

 
class DeepFakeImageDataset(Dataset):

    def __init__(self,
                 data_file="",
                 mode="train",
                 transform=None,
                 img_size=256,):
        super().__init__()

        self.data = read_annotations(data_file)
        self.mode = mode
        self.transform = transform
        self.normalize = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
        self.size = img_size

    def load_train_sample(self, img_info):
        img_path = img_info[0]
        lab = img_info[1]
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB channel
            if self.transform:
                data = self.transform(image=img)
                img = data['image']
            img = img_to_tensor(img, self.normalize)
            return lab, img

        except Exception as e:
            print(img_path, ' error!', e)
            return torch.randn((3, self.size, self.size)), torch.randn((3, self.size, self.size)), 0

    def load_val_sample(self, img_info):
        # real label=0, fake label=1
        img_path = img_info[0]
        lab = img_info[1]
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB chann
            if self.transform:
                data = self.transform(image=img)
                img = data['image']
            img = img_to_tensor(img, self.normalize)
            return lab, img
        except Exception as e:
            print(img_info[0], ' error!', e)
            return torch.randn((3, self.size, self.size)), 0

    def __getitem__(self, index):
        img_info = self.data[index]
        # print(img_info)
        if self.mode == 'train':
            lab, img = self.load_train_sample(img_info)
            return lab, img, img_info[0]
        else:
            lab, img = self.load_val_sample(img_info)
            return lab, img, img_info[0]

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
    train_pos_data_path= '/raid/lpy/data/my_annotations/test.txt'
    train_neg_data_path= '/raid/lpy/data/my_annotations/small_train_fake.txt'
    val_data_path= '/raid/lpy/data/my_annotations/small_val.txt'
    image_size= 256

    dataset = DeepFakeImageDataset(data_file=train_pos_data_path,
                 mode="train",
                 transform=create_transform_resize(256),
                 img_size=256,)
    print(len(dataset))
    dataset[0]