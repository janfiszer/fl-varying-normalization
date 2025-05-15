import logging
import os
import sys
import pickle
import torch
import importlib

from configs import config, enums
from src.deep_learning import metrics, datasets, models
from torch.utils.data import DataLoader

from src.utils.files_operations import get_youngest_dir


def perform_evaluate(test_dir, model_path):
    model_dir = os.path.dirname(model_path)
    representative_test_dir = get_youngest_dir(test_dir)

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

    save_preds_dir = os.path.join(model_dir, "preds", representative_test_dir)
    eval_path = os.path.join(model_dir, "eval", representative_test_dir)

    return unet.evaluate(testloader,
                         compute_std=True,
                         save_preds_dir=save_preds_dir)
                        #  plots_path=os.path.join(model_dir, "eval_visualization", representative_test_dir))


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
    representative_test_dir = get_youngest_dir(test_dir)

    logging.info(f"Model dir is: {model_dir}")

    # the evaluate
    metrics_values, stds = perform_evaluate(test_dir, model_path)

    logging.info(f"metrics: {metrics_values}")
    logging.info(f"stds: {stds}")


    # create the filenames for saving the evaluations results
    descriptive_metric = 'val_gen_dice'
    try:
        metric_filename = f"metrics_{representative_test_dir}_dice_{metrics_values[descriptive_metric]:.2f}.pkl"
        std_filename = f"std_{representative_test_dir}_dice_{metrics_values[descriptive_metric]:.2f}.pkl"
    except KeyError:
        # in case the descriptive_metric wasn't found
        logging.error(f"The provided key ({descriptive_metric}) in the `metrics_values` doesn't existing taking `val_loss` as the key")
        descriptive_metric = 'val_loss'
        
        metric_filename = f"metrics_{representative_test_dir}_loss_{metrics_values[descriptive_metric]:.2f}.pkl"
        std_filename = f"std_{representative_test_dir}_loss_{metrics_values[descriptive_metric]:.2f}.pkl"

    # save the evaluation results     
    metric_filepath = os.path.join(model_dir, metric_filename)
    std_filepath = os.path.join(model_dir, std_filename)

    with open(metric_filepath, "wb") as file:
        pickle.dump(metrics_values, file)
    logging.info(f"Metrics saved to : {metric_filepath}")

    with open(std_filepath, "wb") as file:
        pickle.dump(stds, file)
    logging.info(f"Standard deviations saved to : {std_filepath}")
