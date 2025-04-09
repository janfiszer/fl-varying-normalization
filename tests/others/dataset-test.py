import sys
from os import path
import torch

from configs import config
from src.utils.visualization import plot_pred_tigth
from src.deep_learning.datasets import SegmentationDataset2DSlices

from torch.utils.data import DataLoader

if __name__ == '__main__':
    if config.LOCAL:
        data_dir_ares = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\train"
    else:
        data_dir_ares = sys.argv[1]

    dataset = SegmentationDataset2DSlices(data_dir_ares, ["t1", "t2", "flair"], "mask")

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
                    print(f"WARNING: The shapes are different", image_shape, "!=", image.shape)
                elif target.shape != target_shape:
                    print(f"WARNING: The shapes are different", target_shape, "!=", target.shape)
                else:
                    print("good")

    target_mask = []
    t1_images = []
    t2_images = []
    flair_images = []

    for image, target in zip(images, targets):
        t1_images.append(torch.unsqueeze(image[:, 0], 0))
        t2_images.append(torch.unsqueeze(image[:, 1], 0))
        flair_images.append(torch.unsqueeze(image[:, 2], 0))
        target_mask.append(torch.unsqueeze(target, 0))

    plot_pred_tigth([t1_images, t2_images, flair_images, target_mask], savepath="last_batch_dataset.jpg")



