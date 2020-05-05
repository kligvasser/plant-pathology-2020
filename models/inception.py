import torchvision
import torch.nn as nn

__all__ = ['inception']

class Inception(nn.Module):
    def __init__(self, num_classes=4, two_stage=False):
        super().__init__()

        self.inception = torchvision.models.inception_v3(pretrained=True, num_classes=num_classes)

        if two_stage:
            self.inception.fc = nn.Sequential(nn.Linear(self.inception.fc.in_features, 1024),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(p=0.2),
                                              nn.Linear(1024, num_classes))
        else:
            self.inception.fc = nn.Sequential(nn.Dropout(p=0.2),
                                           nn.Linear(self.inception.fc.in_features, num_classes))

    def forward(self, x):
        x = self.inception(x)
        return x

def inception(**config):
    config.setdefault('num_classes', 4)
    config.setdefault('two_stage', False)

    return Inception(**config)