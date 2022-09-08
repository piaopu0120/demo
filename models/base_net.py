import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


MODEL_DICTS = {}
MODEL_DICTS.update(timm.models.__dict__)


class BaseNet(nn.Module):
    # pretrained=False, drop_rate=0.2,num_classes=2, **kwargs
    def __init__(self, model_name, conf, num_classes=2, **kwargs) -> None:
        super().__init__()
        self.encoder = MODEL_DICTS[model_name](
            pretrained=conf.pretrained, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            self.num_features = self.encoder.last_channel

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(conf.drop_rate)
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        featuremap = self.encoder.forward_features(x)
        x = self.global_pool(featuremap).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    name = "resnet50"
    device = 'cpu'
    model = BaseNet(name)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = torch.rand(4, 3, 224, 224)
        inputs = inputs.to(device)
        out = model(inputs)
        print(out.shape)
