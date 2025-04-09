import logging
import sys
from os import path
import torch

from configs import config
from src.utils.visualization import plot_pred_tigth, plot_all_modalities_and_target
from src.deep_learning.datasets import SegmentationDataset2DSlices

from torch.utils.data import DataLoader

if __name__ == '__main__':
    if config.LOCAL:
        data_dir_ares = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\validation"
    else:
        data_dir_ares = sys.argv[1]

    dataset = SegmentationDataset2DSlices(data_dir_ares, ["t1", "t2", "flair"], "mask", binarize_mask=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    first_iteration = True
    for images, targets in dataloader:
        for image, target in zip(images, targets):
            if first_iteration:
                image_shape = image.shape
                target_shape = target.shape
                first_iteration = False
            else:
                if image.shape != image_shape:
                    logging.log(logging.WARNING, f"The shapes are different", image_shape, "!=", image.shape)
                elif target.shape != target_shape:
                    logging.log(logging.WARNING, f"The shapes are different", target_shape, "!=", target.shape)
                else:
                    logging.log(logging.INFO, "The shapes are matching.")

    plot_all_modalities_and_target(images, targets, predictions_list=targets, column_names=["t1", "t2", "flair", "mask"], rotate_deg=270)



