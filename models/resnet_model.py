import torch
import torch.nn as nn
import torchvision
class ResNet(nn.Module):
    def __init__(self,pretrained=True,numclasses=2) -> None:
        super(ResNet,self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.backbone.fc = torch.nn.Linear(2048,numclasses)
    def forward(self,x):
        return self.backbone(x)
if __name__ == '__main__':
    model = ResNet()
    print(model)