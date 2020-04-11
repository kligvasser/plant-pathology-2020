import torchvision
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet']

class ResNet(nn.Module):
    def __init__(self, num_classes=4, depth=18):
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

        self.classifer = nn.Linear(self.backbone.fc.in_features, num_classes)

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
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.classifer(x)

        return x

def resnet(**config):
    config.setdefault('depth', 18)
    config.setdefault('num_classes', 4)

    return ResNet(**config)