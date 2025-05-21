import logging
import os
import sys
from configs import config
from src.deep_learning import metrics, datasets
from src.utils import visualization
from torch.utils.data import DataLoader

if __name__ == '__main__':
    if config.LOCAL:
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\fl-varying-norm\\3d_dice_test"
        target_dir = os.path.join(data_dir, "targets")
        predicted_dir = os.path.join(data_dir, "preds")
    else:
        target_dir = sys.argv[1]
        predicted_dir = sys.argv[2]

    representative_pred_dir = os.path.basename(predicted_dir)

    # initialing 3d torch dataset
    histogram_suffix = "histogram"
    eval_dataset = datasets.VolumeEvaluation(target_dir, predicted_dir)
    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logging.info(f"Targets loaded from: {target_dir}")
    logging.info(f"Predictions loaded from: {predicted_dir}")

    # initializing metrics
    generalized_dice = metrics.GeneralizedTwoClassDice()
    smoothed_dice = metrics.BinaryDice(binarize_threshold=0.5)
    jaccard_index = metrics.JaccardIndex(binarize_threshold=0.5)

    dice_scores = []
    metrics_objects = {"gen_dice": generalized_dice,
                       "binarized_smoothed_dice": smoothed_dice,
                       "binarized_jaccard_index": jaccard_index,
                       }

    metrics_scores = {metric_name: [] for metric_name in metrics_objects.keys()}

    for batch_index, batch in enumerate(dataloader):
        targets_cpu, predicted_cpu = batch[0], batch[1]

        targets = targets_cpu
        predicted = predicted_cpu

        logging.debug(f"Target pixel sum {targets.sum()}")
        logging.debug(f"Predicted pixel sum {predicted.sum()}")
        logging.debug(f"Predicted target union pixel sum {(predicted * targets).sum()}")

        for metric_name, metrics_obj in metrics_objects.items():
            metric_value = metrics_obj(predicted, targets)
            metrics_scores[metric_name].append(metric_value)
            logging.info(f"{metric_name}: {metric_value}")

    averaged_metrics, stds = metrics.compute_average_std_metric(metrics_scores)
    visualization.plot_distribution(metrics_scores,
                                    os.path.join(predicted_dir, os.path.join(predicted_dir, histogram_suffix)))

    # logging results
    logging.info(f"metrics: {averaged_metrics}")
    logging.info(f"stds: {stds}")

    # saving metrics
    metrics.save_metrics_and_std(averaged_metrics, predicted_dir, stds, representative_pred_dir, descriptive_metric='gen_dice')

