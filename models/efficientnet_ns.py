import torch
import torch.nn as nn

__all__ = ['efficientnet_ns']

class EfficientNet(nn.Module):
    def __init__(self, num_classes=4, b_type=7, two_stage=False):
        super().__init__()
        self.efficientnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b{}_ns'.format(b_type) , pretrained=True)
        in_features = self.efficientnet.classifier.in_features
        if two_stage:
            self.efficientnet.classifier = nn.Sequential(nn.Linear(in_features, 1024),
                                                  nn.ReLU(inplace=True),
                                                  nn.Dropout(p=0.3),
                                                  nn.Linear(1024, num_classes))
        else:
            self.efficientnet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

def efficientnet_ns(**config):
    config.setdefault('b_type', 3)
    config.setdefault('num_classes', 4)
    config.setdefault('two_stage', False)

    return EfficientNet(**config)