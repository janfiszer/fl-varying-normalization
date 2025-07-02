from typing import Dict, List, Any, Optional, Sequence, Tuple, Union, Literal

from torchmetrics.metric import Metric
from torchmetrics.segmentation import GeneralizedDiceScore
import torch
import logging
import pickle
import os
import numpy as np


def metrics_to_str(metrics: Dict[str, List[float]], starting_symbol: str = "", sep="\t"):
    metrics_epoch_str = starting_symbol
    for metric_name, epoch_value in metrics.items():
        metrics_epoch_str += f"{metric_name}: {epoch_value:.3f}{sep}"

    return metrics_epoch_str


class LossGeneralizedTwoClassDice(torch.nn.Module):
    def __init__(self, device: str, binary_crossentropy: bool = False):
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
        if self.binary_crossentropy:
            return f"LossGeneralizedTwoClassDice with BCE"
        else:
            return "LossGeneralizedTwoClassDice"


class LossGeneralizedMultiClassDice(torch.nn.Module):
    def __init__(self, num_classes, device: str, binary_crossentropy: bool = False):
        super(LossGeneralizedMultiClassDice, self).__init__()
        self.dice = GeneralizedDiceScore(num_classes=num_classes, include_background=True).to(device)
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
        if self.binary_crossentropy:
            return f"LossGeneralizedMultiClassDice with BCE"
        else:
            return "LossGeneralizedMultiClassDice"


class GeneralizedMultiClassDice(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        raise NotImplementedError
        super().__init__(**kwargs)

        self.add_state("dice_numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dice_denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape

        numerator, denominator = self.compute_dice_components(preds, targets)
        self.dice_numerator += numerator
        self.dice_denominator += denominator

    def compute(self) -> torch.Tensor:
        """Compute the final generalized dice score."""
        return 2 * self.dice_numerator / self.dice_denominator

    @staticmethod
    def compute_dice_components(preds, targets):
        num_samples_0 = (targets == 0).sum().item()
        num_samples_1 = (targets == 1).sum().item()

        weight_0 = 0 if num_samples_0 == 0 else 1 / (num_samples_0 ** 2)
        weight_1 = 0 if num_samples_1 == 0 else 1 / (num_samples_1 ** 2)

        numerator = weight_1 * (preds * targets).sum() + weight_0 * ((1 - preds) * (1 - targets)).sum()
        denominator = weight_1 * (preds + targets).sum() + weight_0 * ((1 - preds) + (1 - targets)).sum()

        return numerator, denominator

    def __repr__(self):
        return f"GeneralizedTwoClassDice"



class GeneralizedTwoClassDice(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("dice_numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dice_denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape

        numerator, denominator = self.compute_dice_components(preds, targets)
        self.dice_numerator += numerator
        self.dice_denominator += denominator

    def compute(self) -> torch.Tensor:
        """Compute the final generalized dice score."""
        return 2 * self.dice_numerator / self.dice_denominator

    @staticmethod
    def compute_dice_components(preds, targets):
        num_samples_0 = (targets == 0).sum().item()
        num_samples_1 = (targets == 1).sum().item()

        weight_0 = 0 if num_samples_0 == 0 else 1 / (num_samples_0 ** 2)
        weight_1 = 0 if num_samples_1 == 0 else 1 / (num_samples_1 ** 2)

        numerator = weight_1 * (preds * targets).sum() + weight_0 * ((1 - preds) * (1 - targets)).sum()
        denominator = weight_1 * (preds + targets).sum() + weight_0 * ((1 - preds) + (1 - targets)).sum()

        return numerator, denominator

    def __repr__(self):
        return f"GeneralizedTwoClassDice"


class BinaryDice(Metric):

    def __init__(self, smooth=1.0, binarize_threshold=None):
        super(BinaryDice, self).__init__()
        self.add_state("dice_numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dice_denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.smooth = smooth
        self.binarize_threshold = binarize_threshold

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape

        numerator, denominator = self.compute_dice_components(preds, targets, self.binarize_threshold)
        self.dice_numerator += numerator
        self.dice_denominator += denominator

    def compute(self) -> torch.Tensor:
        """Compute the final generalized dice score."""
        return self._compute_smoothed_dice(self.dice_numerator, self.dice_denominator, self.smooth)

    def __repr__(self):
        return f"BinaryDice(smooth={self.smooth}, binarize_threshold={self.binarize_threshold})"

    @staticmethod
    def _compute_smoothed_dice(dice_numerator, dice_denominator, smooth):
        return (2 * dice_numerator + smooth) / (dice_denominator + smooth)

    @staticmethod
    def compute_dice_components(predict, target, binarize_threshold=None):
        if binarize_threshold:
            predict = torch.where(predict > binarize_threshold, 1, 0)

        intersection = torch.sum(predict * target)
        denominator = torch.sum(predict) + torch.sum(target)

        return intersection, denominator


class JaccardIndex(Metric):
    # TODO: How ssim made it work with this
    # full_state_update: bool = False

    def __init__(self, smooth=1.0, binarize_threshold=None):
        super(JaccardIndex, self).__init__()
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.smooth = smooth
        self.binarize_threshold = binarize_threshold

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape

        intersection, denominator = self.compute_jaccard_components(preds, targets, self.binarize_threshold)
        self.intersection += intersection
        self.union += denominator - intersection

    def compute(self) -> torch.Tensor:
        """Compute the final generalized dice score."""
        return self._compute_jaccard(self.intersection, self.union, self.smooth)

    @staticmethod
    def _compute_jaccard(intersection, union, smooth):
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def compute_jaccard_components(predict, target, binarize_threshold):
        if binarize_threshold:
            predict = torch.where(predict > binarize_threshold, 1, 0)

        intersection = torch.sum(predict * target)
        denominator = torch.sum(predict) + torch.sum(target)

        return intersection, denominator

    def __repr__(self):
        return f"BinaryDice(smooth={self.smooth}, binarize_threshold={self.binarize_threshold})"


def compute_average_std_metric(metrics_values: Dict[str, List[float]]) -> Tuple[Dict, Dict]:
    """
    Computes the average for each of the metrics using the sum and the number of steps.
    """
    averaged_metrics = {}
    std_metrics = {}

    for metric_name, metric_values in metrics_values.items():
        numpy_metrics_values = np.array(metric_values)
        averaged_metrics[metric_name] = numpy_metrics_values.sum() / len(metric_values)
        std_metrics[metric_name] = numpy_metrics_values.std()

    return averaged_metrics, std_metrics


def save_metrics_and_std(averaged_metrics, predicted_dir, stds=None, filename_prefix="", descriptive_metric='gen_dice'):
    # create the filenames for saving the evaluations results
    try:
        descriptive_metric_value = averaged_metrics[descriptive_metric]
    except KeyError:
        # in case the descriptive_metric wasn't found
        fixed_descriptive_metric = 'gen_dice'
        logging.error(
            f"The provided key (`{descriptive_metric}`) in the `metrics_values` doesn't existing taking `{fixed_descriptive_metric}` as the key")
        descriptive_metric = fixed_descriptive_metric
        descriptive_metric_value = averaged_metrics[descriptive_metric]

    metric_filename = f"metrics_{filename_prefix}_{descriptive_metric}_{descriptive_metric_value:.2f}.pkl"
    std_filename = f"std_{filename_prefix}_{descriptive_metric}_{descriptive_metric_value:.2f}.pkl"

    # save the evaluation results
    metric_filepath = os.path.join(predicted_dir, metric_filename)
    with open(metric_filepath, "wb") as file:
        pickle.dump(averaged_metrics, file)
    logging.info(f"Metrics saved to : {metric_filepath}")

    # save the stds results
    if stds:
        std_filepath = os.path.join(predicted_dir, std_filename)
        with open(std_filepath, "wb") as file:
            pickle.dump(stds, file)
        logging.info(f"Standard deviations saved to : {std_filepath}")

####################
# OLD SEGMENTATION #
####################


def false_positive_ratio(preds, target):
    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    fp = (false_pred * pos_pred).sum().item()

    tn = (true_pred * neg_pred).sum().item()
    # Compute the false positive ratio
    return fp / (tn + fp) if tn > 0 else 0
