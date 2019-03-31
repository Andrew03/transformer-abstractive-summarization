import torch
from torch import nn

class LMLoss(nn.Module):

    def __init__(self, lm_criterion, opt=None):
        super(LMLoss, self).__init__()
        self.lm_criterion = lm_criterion
        self.opt = opt

    def forward(self, lm_logits, X, mask):
        x_shifted = X[:, 1:, 0].contiguous().view(-1)
        mask = mask[:, 1:].view(-1, mask.size(-1) - 1).float()
        lm_logits = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_losses = self.lm_criterion(lm_logits, x_shifted)
        lm_losses = lm_losses.view(X.size(0), X.size(1) - 1)
        lm_losses = lm_losses * mask
        lm_losses = lm_losses.sum(1) / torch.sum(mask, 1)
        return lm_losses

class SummaryLoss(nn.Module):

    def __init__(self, lm_criterion, opt=None):
        super(SummaryLoss, self).__init__()
        self.lm_criterion = lm_criterion
        self.opt = opt

    def forward(self, lm_logits, X, mask):
        x_shifted = X[:, 1+512+1:, 0].contiguous().view(-1)
        mask = mask[:, 1+512+1:].view(-1, mask.size(-1) - 514).float()
        lm_logits = lm_logits[:, 1+512:-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_losses = self.lm_criterion(lm_logits, x_shifted)
        lm_losses = lm_losses.view(X.size(0), -1)
        lm_losses = lm_losses * mask
        lm_losses = lm_losses.sum(1) / torch.sum(mask, 1)
        return lm_losses
