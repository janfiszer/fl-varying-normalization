import logging
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.files_operations import load_nii_slices, get_nii_filepaths, TransformVolumesToNumpySlices, trim_image


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

    def __init__(self, data_path: str, modalities_names: List, mask_dir: str, image_size=None, binarize_mask=False):
        # declaring booleans
        self.binarize_mask = binarize_mask
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.modalities_names = modalities_names

        self.modalities_filepaths, self.target_filepaths = self.load_full_paths(data_path)

    def __len__(self):
        return len(self.target_filepaths)

    def load_full_paths(self, data_path):
        modalities_filepaths = {modality: [] for modality in self.modalities_names}
        target_filepaths = []

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

        if self.image_size is not None:
            np_image = self._trim_image(np_image)
            np_target = self._trim_image(np_target)

        tensor_image = torch.from_numpy(np.expand_dims(np_image, axis=0))
        tensor_target = torch.from_numpy(np.expand_dims(np_target, axis=0))

        if self.binarize_mask:
            tensor_target = tensor_target > 0
            tensor_target = tensor_target.int()

        # converting to float to be able to perform tensor multiplication
        # otherwise an error
        return tensor_image.float(), tensor_target.float()


class VolumeEvaluation(Dataset):
    def __init__(self, ground_truth_path: str, predicted_path: str, squeeze_pred=True):
        self.ground_truth_path = ground_truth_path
        self.predicted_path = predicted_path
        self.squeeze_pred = squeeze_pred

        patient_ids = [files.split('-')[1] for files in os.listdir(ground_truth_path)]   # the second part of the file is the patients id
        self.patient_ids = list(set(patient_ids))

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]

        print(f"Patient id: {patient_id}")

        target_img_paths = glob(os.path.join(self.ground_truth_path, f"*{patient_id}*{TransformVolumesToNumpySlices.SLICES_FILE_FORMAT}"))
        predicted_img_paths = glob(os.path.join(self.predicted_path, f"*{patient_id}*{TransformVolumesToNumpySlices.SLICES_FILE_FORMAT}"))

        target_slices = [np.load(fp) for fp in target_img_paths]
        predicted_slices = [np.load(fp) for fp in predicted_img_paths]

        if self.squeeze_pred:
            predicted_slices = [pred[0] for pred in predicted_slices]

        target_volume = np.stack(target_slices)
        predicted_volume = np.stack(predicted_slices)

        target_volume = target_volume > 0  # binarize

        return target_volume, predicted_volume
    

class MRIDatasetNII(Dataset):
    """
    A dataset which loads full 3D images .nii images to memory. Might be less efficient and limits some shuffling
    possibilities which is important in training on smaller datasets.
    Therefore, often utilized pipeline was:
    1. TransformNIIDataToNumpySlices to transform the dataset into 2D slices
    2. Then MRIDatasetNumpySlices used as the torch.Dataset to load the data efficiently
    """
    MIN_SLICE_INDEX = 50
    MAX_SLICE_INDEX = 125
    STEP = 1

    def __init__(self, data_dir: str, t1_filepath_from_data_dir, t2_filepath_from_data_dir, transform,
                 image_size=None, transform_order=None, n_patients=1, t1_to_t2=True):
        self.data_dir = data_dir
        self.transform = transform

        if t1_to_t2:
            images_filepaths, targets_filepaths = get_nii_filepaths(data_dir, t1_filepath_from_data_dir,
                                                                    t2_filepath_from_data_dir, n_patients)
        else:
            targets_filepaths, images_filepaths = get_nii_filepaths(data_dir, t1_filepath_from_data_dir,
                                                                    t2_filepath_from_data_dir, n_patients)

        self.images, self.targets = [], []

        for img_path in images_filepaths:
            slices, _, _ = load_nii_slices(img_path, transform_order, image_size, self.MIN_SLICE_INDEX,
                                           self.MAX_SLICE_INDEX)
            self.images.extend(slices)

        for target_path in targets_filepaths:
            slices, _, _ = load_nii_slices(target_path, transform_order, image_size, self.MIN_SLICE_INDEX,
                                           self.MAX_SLICE_INDEX)
            self.targets.extend(slices)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(np.expand_dims(self.images[index], axis=0))
        target = torch.from_numpy(np.expand_dims(self.targets[index], axis=0))

        if self.transform is not None:
            image = self.transform(image)
            target = self.transform(target)

        return image.float(), target.float()
