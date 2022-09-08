import numpy as np
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.efficientnet_pytorch import EfficientNet

__all__ = ['BinaryClassifier']

# MODEL_DICTS = {}
# MODEL_DICTS.update(timm.models.__dict__)


class BinaryClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2, drop_rate=0.2, has_feature=False, pretrained=False, pretrained_path='') -> None:
        """Base binary classifier
        Args:
            encoder ([nn.Module]): Backbone of the DCL
            num_classes (int, optional): Defaults to 2.
            drop_rate (float, optional):  Defaults to 0.2.
            has_feature (bool, optional): Wthether to return feature maps. Defaults to False.
            pretrained (bool, optional): Whether to use a pretrained model. Defaults to False.
        """
        super().__init__()
        self.pretrained_path = pretrained_path
        # self.encoder = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)
        self.encoder = EfficientNet.from_name(encoder, include_top=False)

        # for k, v in kwargs.items():
        #     setattr(self, k, v)

        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            self.num_features = self.encoder.out_channels

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.has_feature = has_feature
        self.feature_squeeze = nn.Conv2d(self.num_features, 1, 1)
        self.fc = nn.Linear(self.num_features, num_classes)

        if pretrained:
            self.load_pretrained_checkpoint()

    def load_pretrained_checkpoint(self):
        print("=> loading pretrained checkpoint '{}'".format(self.pretrained_path))
        checkpoint = torch.load(self.pretrained_path, map_location=torch.device('cpu'))
        # state_dict = checkpoint.get("state_dict", checkpoint)
        # # state_dict.pop('_fc.weight')
        # # state_dict.pop('_fc.bias')
        # # self.encoder.load_state_dict(state_dict)
        #
        # model_state_dict = self.encoder.state_dict()
        #
        # for k, v in state_dict.items():
        #     name = k.replace('module.', '')  # remove `module.`
        #     if 'classifier' in name:
        #         continue
        #     model_state_dict[name] = v
        # self.encoder.load_state_dict(model_state_dict)
        state_dict = checkpoint.get("state_dict", checkpoint)
        i, j = 0, 0
        model_state_dict = self.encoder.state_dict()
        model_state_dict_keys = list(model_state_dict.keys())
        state_dict_keys = list(state_dict.keys())
        out_state_dict = {}
        while (i < len(state_dict_keys)) and (j < len(model_state_dict_keys)):
            shape_i = state_dict[state_dict_keys[i]].shape
            shape_j = model_state_dict[model_state_dict_keys[j]].shape
            if shape_i == shape_j:
                out_state_dict[model_state_dict_keys[j]] = state_dict[state_dict_keys[i]]
                i += 1
                j += 1
            else:
                i += 1
        self.encoder.load_state_dict(out_state_dict, strict=True)


    def forward(self, x):
        featuremap = self.encoder.extract_features(x)
        x = self.global_pool(featuremap).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.has_feature:
            return x, featuremap
        return x


if __name__ == '__main__':
    name = "efficientnet-b4"
    device = 'cpu'
    model = BinaryClassifier(name, pretrained=True, )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = torch.rand(4, 3, 224, 224)
        inputs = inputs.to(device)
        out = model(inputs)
        print(out.shape)
