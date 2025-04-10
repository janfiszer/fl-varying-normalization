import logging
import sys
from os import path
from pathlib import Path

from configs import config
from src.utils.visualization import plot_all_modalities_and_target
from src.deep_learning.datasets import SegmentationDataset2DSlices

from torch.utils.data import DataLoader

if __name__ == '__main__':
    if config.LOCAL:
        data_dir_ares = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\validation"
    else:
        data_dir_ares = sys.argv[1]
    visualization_dir = "batches_visualization"
    visualization_path = path.join(data_dir_ares, visualization_dir)
    Path(visualization_path).mkdir(exist_ok=True)

    dataset = SegmentationDataset2DSlices(data_dir_ares, config.USED_MODALITIES, config.MASK_DIR, binarize_mask=True)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    for i, (images, targets) in enumerate(dataloader):
        for image, target in zip(images, targets):
            # getting the shapes in the first itteration
            if i == 0:
                image_shape = image.shape
                target_shape = target.shape
            else:
                if image.shape != image_shape:
                    logging.log(logging.ERROR, f"The shapes are different", image_shape, "!=", image.shape)
                elif target.shape != target_shape:
                    logging.log(logging.ERROR, f"The shapes are different", target_shape, "!=", target.shape)
                else:
                    logging.log(logging.DEBUG, "The shapes are matching.")
        plot_all_modalities_and_target(images, targets, column_names=config.USED_MODALITIES + [config.MASK_DIR], rotate_deg=270, savepath=path.join(visualization_path, f"batch{i}_dataset_test.jpg"))

    logging.log(logging.INFO, "Testing process: ENDED")

    # plot_all_modalities_and_target(images, targets, column_names=["t1", "t2", "flair", "mask"], rotate_deg=270, savepath=path.join(data_dir_ares, "last_batch_dataset_test.jpg"))



