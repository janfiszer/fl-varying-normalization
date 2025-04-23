from typing import Dict, List, Any, Optional, Sequence, Tuple, Union, Literal

from torch import Tensor
from torchmetrics.functional.segmentation.generalized_dice import _generalized_dice_validate_args, \
    _generalized_dice_update, _generalized_dice_compute
from torchmetrics.image import StructuralSimilarityIndexMeasure

from configs import config
from torchmetrics.metric import Metric
from torchmetrics.segmentation import GeneralizedDiceScore

import torch


def metrics_to_str(metrics: Dict[str, List[float]], starting_symbol: str = "", sep="\t"):
    metrics_epoch_str = starting_symbol
    for metric_name, epoch_value in metrics.items():
        metrics_epoch_str += f"{metric_name}: {epoch_value:.3f}{sep}"

    return metrics_epoch_str


class GeneralizedDiceLoss(torch.nn.Module):
    def __init__(self, num_classes: int, device: str, binary_crossentropy: bool = False):
        super(GeneralizedDiceLoss, self).__init__()
        self.dice = GeneralizedDiceScore(num_classes, per_class=True).to(device)
        self.binary_crossentropy = binary_crossentropy

        if binary_crossentropy:
            self.bce_loss = torch.nn.BCELoss().to(device)

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


class LossGeneralizedTwoClassDice(torch.nn.Module):
    def __init__(self, device, binary_crossentropy: bool = False):
        super(LossGeneralizedTwoClassDice, self).__init__()
        self.dice = GeneralizedTwoClassDice().to(device)
        self.binary_crossentropy = binary_crossentropy

        if binary_crossentropy:
            self.bce_loss = torch.nn.BCELoss()

    def forward(self, predict, target):
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


class GeneralizedTwoClassDice(Metric):
    def __init__(self, binarize_threshold: int = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.add_state("dice_score", default=torch.tensor(0.0), dist_reduce_fx="cat")
        self.add_state("samples", default=torch.tensor(0), dist_reduce_fx="sum")

        # self.add_state("dice_score_no_binarized", default=torch.tensor(0.0), dist_reduce_fx="cat")
        # self.register_buffer("device_helper", torch.tensor(0.))
        # self.register_buffer("binarize_threshold", torch.tensor(binarize_threshold))
        # self.binarize_threshold = torch.tensor(binarize_threshold).to(device) # TODO: reconsider where is should be moved to the device

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape

        # cast_1 = (preds > self.binarize_threshold).to(dtype=torch.float64)
        # cast_2 = (preds > self.binarize_threshold).float()
        # cast_3 = (preds > self.binarize_threshold).type(torch.float64)
        # cast_4 = preds.int()

        self.dice_score = self.compute_dice(preds, targets)

        # print(f"preds: {preds.get_device()}")
        # print(f"self.binarize_threshold: {self.binarize_threshold.get_device()}")
        # print(f"targets: {targets.get_device()}")
        # print(f"cast_1: {cast_1.get_device()}")
        # print(f"cast_2: {cast_2.get_device()}")
        # print(f"cast_3: {cast_3.get_device()}")
        # print(f"cast_4: {cast_4.get_device()}")
        # print(f"dice_score_no_binarized: {dice_score_no_binarized.get_device()}")

        # for i, cast in enumerate([cast_1, cast_2, cast_3, cast_4]):
        #     try:
        #         self.dice_score += self.compute_dice(cast, targets)
        #     except RuntimeError:
        #         print(f"cast {i} sucks")
        self.samples += preds.shape[0]

    def compute(self) -> torch.Tensor:
        """Compute the final generalized dice score."""
        return self.dice_score / self.samples

    def reset(self):
        self.dice_score = torch.tensor(0.0)
        self.samples = torch.tensor(0)

    def compute_dice(self, preds, targets):
        num_samples_0 = (targets == 0).sum().item()
        num_samples_1 = (targets == 1).sum().item()

        weight_0 = 0 if num_samples_0 == 0 else 1 / (num_samples_0 ** 2)
        weight_1 = 0 if num_samples_1 == 0 else 1 / (num_samples_1 ** 2)

        # preds_int = preds.int()
        # print("preds_int")
        # print((preds_int == 0).sum().item())
        # print((preds_int == 1).sum().item())
        # print(torch.flatten(preds).shape)

        # preds_thresholded = (preds > self.binarize_threshold).float()
        # print("preds_thresholded")
        # print((preds_thresholded == 0).sum().item())
        # print((preds_thresholded == 1).sum().item())
        # print(torch.flatten(preds).shape)
        intersect = weight_1 * (preds * targets).sum() + weight_0 * ((1 - preds) * (1 - targets)).sum()
        denominator = weight_1 * (preds + targets).sum() + weight_0 * ((1 - preds) + (1 - targets)).sum()

        return 2 * intersect / denominator


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
    loss = 1 - generalized_dice(predict, target)

    return loss


def generalized_dice(predict, target):
    num_samples_0 = (target == 0).sum().item()
    num_samples_1 = (target == 1).sum().item()

    weight_0 = 0 if num_samples_0 == 0 else 1 / (num_samples_0 ** 2)
    weight_1 = 0 if num_samples_1 == 0 else 1 / (num_samples_1 ** 2)

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
