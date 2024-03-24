import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_logit_adjust_loss1(input, target, positive_samples=493, negative_samples=6126, reduction="mean"):
    """Logit adjust loss for binary classification problems."""
    adjusted_input = 1.0 / (((negative_samples / positive_samples) * 1 * (1 / input - 1)) + 1)
    loss = torch.nn.functional.binary_cross_entropy(adjusted_input, target, reduction=reduction)
    return loss


def binary_logit_adjust_loss(input, target, positive_samples=493, negative_samples=6126, reduction="mean"):
    """Logit adjust loss for binary classification problems."""
    sum = positive_samples + negative_samples
    logit_adjustment1 = (negative_samples / sum)
    logit_adjustment2 = (positive_samples / sum)
    adjusted_logits = input + 1.0 * torch.log((1 - target) * logit_adjustment1 + target * logit_adjustment2)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(adjusted_logits, target, reduction=reduction)
    return loss


def binary_logit_adjust_loss2(input, target, positive_samples=493, negative_samples=6126, reduction="mean"):
    """Logit adjust loss for binary classification problems."""
    logit_adjustment = negative_samples / positive_samples
    adjusted_logits = input + torch.log((1 - target) * logit_adjustment + target)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(adjusted_logits, target, reduction=reduction)
    return loss


def binary_focal_loss(input, target, alpha=0.25, gamma=2.0, reduction="mean"):
    """Focal loss for binary classification problems."""
    pt = input * target + (1 - input) * (1 - target)
    alpha_factor = alpha * target + (1 - alpha) * (1 - target)
    modulating_factor = (1.0 - pt) ** gamma
    loss = -alpha_factor * modulating_factor * torch.log(pt)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets, reduction):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if reduction == "mean":
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)
