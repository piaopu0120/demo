import torch
import torch.nn as nn
import torchvision
import numpy as np
class ResNet_dct(nn.Module):
    def __init__(self,pretrained=True,img_size=256,numclasses=2) -> None:
        super(ResNet_dct,self).__init__()
        self._DCT_all = nn.Parameter(torch.tensor(self.DCT_mat(img_size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(self.DCT_mat(img_size)).float(), 0, 1), requires_grad=False) # DCT的转置

        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.backbone.fc = torch.nn.Linear(2048,numclasses)

    def forward(self,x):
        # x = self._DCT_all @ x 
        # x = x @ self._DCT_all_T # freq
        x = self._DCT_all @ x @ self._DCT_all_T
        x = self.backbone(x)
        return x
        
    def DCT_mat(self,size):
        m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
        return m

if __name__ == '__main__':
    model = ResNet_dct()
    print(model)