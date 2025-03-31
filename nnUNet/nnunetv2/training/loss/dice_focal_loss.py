import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Should be a tensor of class weights or a scalar
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: raw logits, shape (B, C, ...)
        targets: ground truth, shape (B, ...)
        """
        # Same as TopKLoss: remove singleton channel if present
        if targets.ndim == 5 and targets.shape[1] == 1:
            targets = targets[:, 0].long()

        alpha_weight = self.alpha.to(inputs.device) if self.alpha is not None else None

        # Alpha can be None, scalar, or tensor of shape [C]
        if self.ignore_index is not None:
            ce_loss = F.cross_entropy(
                inputs, targets,
                weight=alpha_weight,
                reduction='none',
                ignore_index=self.ignore_index
            )
        else:
            ce_loss = F.cross_entropy(
                inputs, targets,
                weight=alpha_weight,
                reduction='none'
            )
            
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DC_and_Focal_Loss(nn.Module):
    def __init__(self, soft_dice_kwargs, gamma=2.0, alpha=None, weight_dice=1.0, weight_ce=1.0, ignore_index=None):
        super(DC_and_Focal_Loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs,) 
        self.ce = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)

    def forward(self, outputs, targets):
        dice_loss = self.dice(outputs, targets)
        focal_loss = self.ce(outputs, targets)
        total_loss = self.weight_dice * dice_loss + self.weight_ce * focal_loss
        return total_loss

