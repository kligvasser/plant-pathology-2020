import torchvision
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet']

class ResNet(nn.Module):
    def __init__(self, num_classes=4, depth=18, two_stage=False):
        super().__init__()

        if depth == 18:
            self.backbone = torchvision.models.resnet18(pretrained=True)
        elif depth == 34:
            self.backbone = torchvision.models.resnet34(pretrained=True)
        elif depth == 50:
            self.backbone = torchvision.models.resnet50(pretrained=True)
        elif depth == 101:
            self.backbone = torchvision.models.resnet101(pretrained=True)
        else:
            self.backbone = torchvision.models.resnet152(pretrained=True)

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

def resnet(**config):
    config.setdefault('depth', 18)
    config.setdefault('num_classes', 4)
    config.setdefault('two_stage', False)

    return ResNet(**config)