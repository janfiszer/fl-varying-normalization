import logging
import os
from typing import List, Tuple, Union

from src.utils.files_operations import TransformVolumesToNumpySlices

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset2DSlices(Dataset):
    """
    Dataset class with previous use of TransformNIIDataToNumpySlices
    Required directory structure:
        `data_dir`/
        │── patient_name/
        │   ├── network input modalities e.g.
        │   ├── t1
        │   ├── t2
        │   ├── flair
        │   ├── `mask_dir`
    """
    EPS = 1e-6

    def __init__(self, data_paths: Union[str, List], modalities_names: List, mask_dir: str,  binarize_mask=False, num_classes=1):
        # declaring booleans
        self.binarize_mask = binarize_mask
        self.num_classes = num_classes
        self.mask_dir = mask_dir
        self.modalities_names = modalities_names

        self.modalities_filepaths, self.target_filepaths = self.load_full_paths(data_paths)

        logging.info(f"Dataset 'SegmentationDataset2DSlices' loaded {len(self.target_filepaths)} filepaths from {data_paths}.")

    def __len__(self):
        return len(self.target_filepaths)

    def load_full_paths(self, data_paths):
        modalities_filepaths = {modality: [] for modality in self.modalities_names}
        target_filepaths = []

        if not isinstance(data_paths, List):
            data_paths = [data_paths]

        for data_path in data_paths:
            for patient_dir in os.listdir(data_path):
                patient_fullpath = os.path.join(data_path, patient_dir)
                for slice_dir in os.listdir(patient_fullpath):
                    slice_dir_fullpath = os.path.join(patient_fullpath, slice_dir)
                    for modality in self.modalities_names:
                        slice_file_fullpath = os.path.join(slice_dir_fullpath, f"{modality}{TransformVolumesToNumpySlices.SLICES_FILE_FORMAT}")
                        modalities_filepaths[modality].append(slice_file_fullpath)
                    target_filepaths.append(os.path.join(slice_dir_fullpath, f"{self.mask_dir}{TransformVolumesToNumpySlices.SLICES_FILE_FORMAT}"))

        # checking if all the lists are the same size
        for modality in self.modalities_names:
            assert (len(target_filepaths) == len(modalities_filepaths[modality]))

        return modalities_filepaths, target_filepaths

    def __getitem__(self, index):
        # loading images
        images_paths = [self.modalities_filepaths[modality_name][index] for modality_name in self.modalities_names]
        np_images = [np.load(image_path) for image_path in images_paths]
        np_image = np.stack(np_images)

        # loading target mask
        np_target = np.load(self.target_filepaths[index])

        tensor_image = torch.from_numpy(np_image)
        tensor_target = torch.from_numpy(np_target)

        # tensor_image = torch.from_numpy(np.expand_dims(np_image, axis=0))

        if self.binarize_mask:
            tensor_target = np.expand_dims(tensor_target, axis=0)
            tensor_target = tensor_target > 0
        else:
            logging.debug(f"tensor_target values are {torch.unique(tensor_target)}")
            tensor_target = torch.nn.functional.one_hot(tensor_target.to(torch.int64), self.num_classes)   # TODO: clip
            tensor_target = tensor_target.permute(2, 0, 1)  # permute to have the shape (n_classes, image_shape)
        # converting to float to be able to perform tensor multiplication
        # otherwise an error
        return tensor_image.float(), tensor_target.int()


class VolumeEvaluation(Dataset):
    def __init__(self, ground_truth_path: str, predicted_path: str, mask_target_filename: str = "mask.npy", squeeze_pred=True, binarize_target: bool = True):
        self.ground_truth_dir_paths = [os.path.join(ground_truth_path, dir_name) for dir_name in sorted(os.listdir(ground_truth_path))]
        self.predicted_dir_paths = [os.path.join(predicted_path, dir_name) for dir_name in sorted(os.listdir(predicted_path))]

        self.mask_target_filename = mask_target_filename
        self.squeeze_pred = squeeze_pred
        self.binarize_target = binarize_target

    def __len__(self):
        return len(self.ground_truth_dir_paths)

    def __getitem__(self, index):
        # getting the patient dir by index
        ground_truth_dir = self.ground_truth_dir_paths[index]
        predicted_dir = self.predicted_dir_paths[index]
        logging.info(f"Patient ground truth is: {ground_truth_dir}, predicted: {predicted_dir}")

        # loading the patient slices (target and predicted)
        target_slices = [np.load(os.path.join(ground_truth_dir, filename, self.mask_target_filename))
                         for filename in sorted(os.listdir(ground_truth_dir))]
        predicted_slices = [np.load(os.path.join(predicted_dir, filename)) for filename in sorted(os.listdir(predicted_dir))]

        # reducing a dimension of the predicted slices
        if self.squeeze_pred:
            predicted_slices = [pred[0] for pred in predicted_slices]

        # stacking the slices
        target_volume = np.stack(target_slices)
        predicted_volume = np.stack(predicted_slices)

        if self.binarize_target:
            target_volume = (target_volume > 0).astype(int)

        return target_volume, predicted_volume
