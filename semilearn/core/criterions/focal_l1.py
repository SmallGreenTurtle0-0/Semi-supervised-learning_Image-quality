import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalL1Loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, beta=.2, activate='sigmoid'):
        super(FocalL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.activate = activate

    def forward(self, inputs, targets, weights=None, reduction='none'):
        loss = F.l1_loss(inputs, targets, reduction='none')
        loss *= (torch.tanh(self.beta * torch.abs(inputs - targets))) ** self.gamma if self.activate == 'tanh' else \
            (2 * torch.sigmoid(self.beta * torch.abs(inputs - targets)) - 1) ** self.gamma
        if weights is not None:
            loss *= weights.expand_as(loss)
        if reduction == 'mean':
            loss = torch.mean(loss)
        return loss
