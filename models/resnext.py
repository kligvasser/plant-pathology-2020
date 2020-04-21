import torchvision
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnext']

class ResNet(nn.Module):
    def __init__(self, num_classes=4, depth=50, two_stage=False):
        super().__init__()

        if depth == 50:
            self.backbone = torchvision.models.resnext50_32x4d(pretrained=True)
        else:
            self.backbone = torchvision.models.resnext101_32x8d(pretrained=True)

        if two_stage:
            self.classifer = nn.Sequential(nn.Linear(self.backbone.fc.in_features, 1024),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(p=0.2),
                                           nn.Linear(1024, num_classes))
        else:
            self.classifer = nn.Sequential(nn.Dropout(p=0.2),
                                           nn.Linear(self.backbone.fc.in_features, num_classes))

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(start_dim=1)
        x = self.classifer(x)

        return x

def resnext(**config):
    config.setdefault('depth', 50)
    config.setdefault('num_classes', 4)
    config.setdefault('two_stage', False)

    return ResNet(**config)