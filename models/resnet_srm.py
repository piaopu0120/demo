import torch
import torch.nn as nn
import torchvision
from .frequency.srm import setup_srm_layer
class ResNet_srm(nn.Module):
    def __init__(self,pretrained=True,numclasses=2) -> None:
        super(ResNet_srm,self).__init__()
        self.srm_conv = setup_srm_layer(input_channels=3)
        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.backbone.fc = torch.nn.Linear(2048,numclasses)
    def forward(self,x):
        x = self.srm_conv(x)
        x = self.backbone(x)
        return x
if __name__ == '__main__':
    model = ResNet_srm()
    print(model)