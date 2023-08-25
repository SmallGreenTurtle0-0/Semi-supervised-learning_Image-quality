import torch
import torch.nn as nn

class MAE_Loss(nn.Module):
    def __init__(self):
        super(MAE_Loss, self).__init__()

    def forward(self, pred, targets, reduction='none'):
        loss = torch.abs(pred - targets)
        if reduction == 'mean':
            return loss.mean()
        else:
            return loss
            