import sys
import os
import logging
import pickle
from configs import config

from exe.evaluations import evaluate
import matplotlib.pyplot as plt


def visulize_relation(metrics_dict):
    all_metrics = list(next(iter(metrics_dict.values())).keys())

    # Step 2: Prepare data for plotting
    scalar_keys = sorted(metrics_dict.keys())
    metric_values = {metric: [metrics_dict[k][metric] for k in scalar_keys] for metric in all_metrics}

    # Step 3: Plot each metric
    for metric in all_metrics:
        logging.info(f"Visualizing metric: {metric}")
        plt.figure()
        plt.plot(scalar_keys, metric_values[metric], marker='o')
        plt.title(f'{metric} vs batch size')
        plt.xlabel('Batch size')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    if config.LOCAL:
        test_dir = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\small"
        model_path = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\trained_models\\model-zscore-MSE_DSSIM-ep2-lr0.001-GN-2025-04-24-8h\\best_model.pth"
    else:
        test_dir = sys.argv[1]
        model_path = sys.argv[2]

    if len(sys.argv) > 4:
        config_path = sys.argv[4]
    else:
        config_path = os.path.join(os.path.join(model_path, test_dir, model_path), "config.py")

    model_dir = os.path.dirname(model_path)

    logging.info(f"Model dir is: {model_dir}")

    batch_size_to_metrics = {}
    batch_size_to_std = {}

    for batch_size in range(1, 25):
        logging.info(f"Evaluating for: {batch_size}")
        metrics = evaluate.perform_evaluate(batch_size, test_dir, model_path)
        batch_size_to_metrics[batch_size] = metrics

    visulize_relation(batch_size_to_metrics)

    with open("metrics_to_batch_size.pkl", "wb") as file:
        pickle.dump(batch_size_to_metrics, file)
