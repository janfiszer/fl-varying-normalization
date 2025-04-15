from typing import Dict, List, Any

import numpy as np
from configs import config
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric
from torchmetrics.segmentation import GeneralizedDiceScore

from scipy import signal

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide


def metrics_to_str(metrics: Dict[str, List[float]], starting_symbol="", sep="\t"):
    metrics_epoch_str = starting_symbol
    for metric_name, epoch_value in metrics.items():
        metrics_epoch_str += f"{metric_name}: {epoch_value:.3f}{sep}"

    return metrics_epoch_str


class GeneralizedDiceLoss(torch.nn.Module):
    def __init__(self, num_classes, device, binary_crossentropy=False):
        super(GeneralizedDiceLoss, self).__init__()
        self.dice = GeneralizedDiceScore(num_classes, per_class=True).to()
        self.binary_crossentropy = binary_crossentropy

        if binary_crossentropy:
            self.bce_loss = torch.nn.BCELoss().to()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        # TODO: dice=0 when no target ALWAYS...
        dice_scores = self.dice(predict, target)
        loss = 1 - dice_scores.mean()

        if self.binary_crossentropy:
            bce_loss = self.bce_loss(predict, target.float())
            total_loss = loss + bce_loss
        else:
            total_loss = loss

        return total_loss

    def __repr__(self):
        return "GeneralizedDiceLoss"


####################
# OLD SEGMENTATION #
####################

class BinaryDiceLoss(torch.nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, device, smooth=1, p=2, binary_crossentropy=False):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.binary_crossentropy = binary_crossentropy

        if binary_crossentropy:
            self.bce_loss = torch.nn.BCELoss().to(device)

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        loss = loss.mean()

        if self.binary_crossentropy:
            bce_loss = self.bce_loss(predict, target.float())
            total_loss = loss + bce_loss
        else:
            total_loss = loss

        return total_loss

    def __repr__(self):
        if self.binary_crossentropy:
            return "Dice with BCE LOSS"
        else:
            return "Dice LOSS"


class BinaryDice(torch.nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDice, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        intersection = torch.sum(predict * target)
        sum_pred_target = torch.sum(predict) + torch.sum(target)

        dice_coefficient = (2 * intersection + self.smooth) / (sum_pred_target + self.smooth)
        return dice_coefficient

    def __repr__(self):
        return "BinaryDice"


def weighted_BCE(predict, target):
    total_samples = torch.numel(target)
    num_samples_0 = (target == 0).sum().item()
    num_samples_1 = (target == 1).sum().item()

    weight_0 = 0 if num_samples_0 == 0 else total_samples / (num_samples_0 * 2)
    weight_1 = 0 if num_samples_1 == 0 else total_samples / (num_samples_1 * 2)

    loss = -(weight_1 * (target * torch.log(predict)) + weight_0 * ((1 - target) * torch.log(1 - predict)))

    return torch.mean(loss)


def loss_generalized_dice(predict, target):
    num_samples_0 = (target == 0).sum().item()
    num_samples_1 = (target == 1).sum().item()

    weight_0 = 0 if num_samples_0 == 0 else 1/(num_samples_0*num_samples_0)
    weight_1 = 0 if num_samples_1 == 0 else 1/(num_samples_1*num_samples_1)

    intersect = weight_1*(predict * target).sum() + weight_0*((1 - predict) * (1 - target)).sum()
    denominator = weight_1*(predict + target).sum() + weight_0*((1 - predict) + (1 - target)).sum()

    loss = 1 - (2*(intersect/denominator))

    return loss


def generalized_dice(predict, target):
    num_samples_0 = (target == 0).sum().item()
    num_samples_1 = (target == 1).sum().item()

    weight_0 = 0 if num_samples_0 == 0 else 1 / (num_samples_0 * num_samples_0)
    weight_1 = 0 if num_samples_1 == 0 else 1 / (num_samples_1 * num_samples_1)

    intersect = weight_1 * (predict * target).sum() + weight_0 * ((1 - predict) * (1 - target)).sum()
    denominator = weight_1 * (predict + target).sum() + weight_0 * ((1 - predict) + (1 - target)).sum()

    return 2 * intersect / denominator

# def loss_generalized_dice(predict, target):
#     dice = generalized_dice(predict, target)
#     return 1 - dice

def dice_2_class(predict, target, eps=1):
    pred_mutl_target = (predict * target).sum()
    pred_plus_target = (predict + target).sum()

    opp_pred_mutl_target = ((1 - predict) * (1 - target)).sum()
    opp_pred_plus_target = ((1 - predict) + (1 - target)).sum()

    ones_faction = (pred_mutl_target + eps)/ (pred_plus_target + eps)
    zeros_faction = (opp_pred_mutl_target + eps) / (opp_pred_plus_target + eps)

    if ones_faction > 0.5:
        # print(f"\t\t\t\tThe one fraction is equal to: {ones_faction}")
        # TODO: not sure how to do it cause to have a good gradient, maybe setting like this is alright IDK
        ones_faction = torch.tensor(0.5)

    # print(f"\t\tNot weighted dice components: {pred_mutl_target+eps}/{pred_plus_target+eps} + {opp_pred_mutl_target+eps}/{opp_pred_plus_target+eps}")

    dice_score = ones_faction + zeros_faction

    if dice_score > 1:
        dice_score = torch.tensor(1.0)
    return dice_score

class LossDice2Class(torch.nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=True):
        super(LossDice2Class, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        loss = 1 - dice_2_class(predict, target, eps=self.smooth)

        return loss

    def __repr__(self):
        return "Domi LOSS"

def loss_dice_2_class(predict, target):
    dice = dice_2_class(predict, target)
    return 1 - dice


class DomiBinaryDiceLoss(torch.nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, weighted=True):
        super(DomiBinaryDiceLoss, self).__init__()
        if weighted:
            self.bce_loss = weighted_BCE
        else:
            self.bce_loss = torch.nn.BCELoss().to(config.DEVICE)

    def forward(self, predict, target):
        loss = self.bce_loss(predict, target) + loss_dice_2_class(predict, target)

        return loss

    def __repr__(self):
        return "Domi LOSS"


def false_positive_ratio(preds, target):
    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    fp = (false_pred * pos_pred).sum().item()

    tn = (true_pred * neg_pred).sum().item()
    # Compute the false positive ratio
    return fp / (tn + fp) if tn > 0 else 0
