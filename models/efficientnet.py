import efficientnet_pytorch
import torch.nn as nn

__all__ = ['efficientnet']

class EfficientNet(nn.Module):
    def __init__(self, num_classes=4, b_type=7, modify_fc=False):
        super().__init__()
        self.efficientnet = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b{}'.format(b_type), num_classes=num_classes)

        if modify_fc:
            in_features = self.efficientnet._fc.in_features
            self.efficientnet._fc = nn.Sequential(nn.Linear(in_features, 512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Dropout(p=0.3),
                                                  nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.efficientnet(x)
        return x

def efficientnet(**config):
    config.setdefault('b_type', 7)
    config.setdefault('num_classes', 4)
    config.setdefault('modify_fc', False)

    return EfficientNet(**config)