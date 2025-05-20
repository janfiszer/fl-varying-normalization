import logging
import os
import sys
import torch

from configs import config, enums
from src.deep_learning import metrics, datasets, models
from torch.utils.data import DataLoader
from pathlib import Path
from src.utils.files_operations import get_youngest_dir


def perform_evaluate(test_dir, model_path, representative_test_dir):
    model_dir = os.path.dirname(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    testset = datasets.SegmentationDataset2DSlices(test_dir,
                                                   modalities_names=config.USED_MODALITIES,
                                                   mask_dir=config.MASK_DIR,
                                                   binarize_mask=True)

    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    criterion = metrics.LossGeneralizedTwoClassDice(device)
    logging.info(f"Taken criterion is: {criterion}")
    unet = models.UNet(criterion, descriptive_metric="gen_dice").to(device)

    try:
        logging.info(f"Loading model from: {model_path}")
        unet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        logging.error(f"You are in {os.getcwd()} and there is no given path")
        exit()

    logging.info(f"Testing on the data from: {test_dir}")

    logging.info(f"Model and data loaded; evaluation starts...")

    save_preds_dir = os.path.join(model_dir, "preds_from_nonorm", representative_test_dir)
    eval_path = os.path.join(model_dir, "eval", representative_test_dir)

    return unet.evaluate(testloader,
                         compute_std=True,
                        #  save_preds_dir=save_preds_dir,
                        #  plots_path=os.path.join(model_dir, "eval_visualization_from_nonorm", representative_test_dir), 
                        #  plot_metrics_distribution=True,
                        #  plot_every_batch_with_metrics=True 
                         )


if __name__ == '__main__':
    # define the data and model paths
    if config.LOCAL:
        test_dir = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\small"
        model_path = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\trained_models\\st\\model-all-ep16-lr0.001-GN-2025-04-30-13h\\best_model.pth"
    else:
        test_dir = sys.argv[1]
        model_path = sys.argv[2]

    # extract directories names from the full paths
    model_dir = os.path.dirname(model_path)
    # representative_test_dir = get_youngest_dir(test_dir)
    unwanted_representative_dir_names = ["train", "test", "validation"]

    representative_test_dir = os.path.basename(test_dir)
    if test_dir in unwanted_representative_dir_names:    
        representative_test_dir = get_youngest_dir(test_dir)

    logging.info(f"Model dir is: {model_dir}")
    logging.info(f"Representative name used for logging and as saving fingerprint is: {representative_test_dir}")

    # the evaluation
    metrics_values, stds = perform_evaluate(test_dir, model_path, representative_test_dir)

    # logging results
    logging.info(f"metrics: {metrics_values}")
    logging.info(f"stds: {stds}")

    # create the directory where they will be stored   
    metric_dir = os.path.join(model_dir, "metrics_from_nonorm")
    Path(metric_dir).mkdir(exist_ok=True)

    # save the evaluation results
    metrics.save_metrics_and_std(metrics_values, metric_dir, stds, descriptive_metric='gen_dice')
