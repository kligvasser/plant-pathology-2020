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
        hot.scatter_(1, labels.unsqeeze(dim=-1), 1.0)
        return hot
