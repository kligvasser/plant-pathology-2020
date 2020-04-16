import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseCrossEntropy(nn.Module):
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, targets):
        logits = logits.float()
        labels = self.hot_one(logits, targets)
        logprobs = F.log_softmax(logits, dim=-1)
        loss = -labels * logprobs
        loss = loss.sum(dim=-1)
        return loss.mean()

    def hot_one(self, logits, labels):
        hot = torch.zeros(size=logits.size()).to(logits.device)
        hot.scatter_(1, labels, 1.0)
        return hot

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1. - pt) ** self.gamma * ce_loss).mean()

        return focal_loss

class CutMixCrossEntropyLoss(torch.nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().to(input.device)
        return cross_entropy(input, target, self.size_average)

def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))