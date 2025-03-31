import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = softmax_helper_dim1(inputs)
        num_classes = inputs.shape[1]
        
        # 💡 Fix shape: if targets shape is [B, 1, D, H, W], squeeze it
        if targets.ndim == 5 and targets.shape[1] == 1:
            targets = targets[:, 0]

        targets = targets.long()
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        TP = (inputs * targets_one_hot).sum(dims)
        FP = (inputs * (1 - targets_one_hot)).sum(dims)
        FN = ((1 - inputs) * targets_one_hot).sum(dims)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky.mean()


class DiceTverskyLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, alpha=0.3, beta=0.7, weight_dice=1.0, weight_tversky=1.0):
        super().__init__()
        self.soft_dice = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.tversky = TverskyLoss(alpha, beta)
        self.weight_dice = weight_dice
        self.weight_tversky = weight_tversky

    def forward(self, inputs, targets):
        dice_loss = self.soft_dice(inputs, targets)
        tversky_loss = self.tversky(inputs, targets)
        return self.weight_dice * dice_loss + self.weight_tversky * tversky_loss

