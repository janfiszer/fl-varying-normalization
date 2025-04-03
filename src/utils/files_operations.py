import logging
import os
import random
import re
import traceback
from glob import glob
from typing import Tuple, Optional, Dict

from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


class TransformNIIDataToNumpySlices:
    # by investigation in eda.ipynb obtained
    # MIN_SLICE_INDEX = 50
    # MAX_SLICE_INDEX = 125
    MIN_SLICE_INDEX = -1
    MAX_SLICE_INDEX = -1
    SLICES_FILE_FORMAT = ".npy"
    DIVISION_SETS = ["train", "test", "validation"]
    """
        Class that enables transform volumes into 2D slices set (train, test, split), 
        which are having substantial part of the brain (`target_zero_ratio`). 
        
        From: 
        `origin_data_dir`/
        │── patient_name_slice/
        │   ├── network input modalities e.g.
        │   ├── t1
        │   ├── t2
        │   ├── flair
        │   ├── `mask_dir`
        
        
        To:
        `target_root_dir`/
        │── train/
        │   ├── patient_name_slice e.g. UCSF_001_slice58
        │   │   ├── network input modalities e.g.
        │   │   ├── t1.npy
        │   │   ├── t2.npy
        │   │   ├── flair.npy
        │   │   ├── `mask_dir`
        │── test/
        │   ├── patient_name_slice e.g. UCSF_002_slice58
        │   │   ├── network input modalities e.g.
        │   │   ├── t1.npy
        │   │   ├── t2.npy
        │   │   ├── flair.npy
        │   │   ├── `mask_dir`
        │── validation/
        │   ├── patient_name_slice e.g. UCSF_003_slice58
        │   │   ├── network input modalities e.g.
        │   │   ├── t1.npy
        │   │   ├── t2.npy
        │   │   ├── flair.npy
        │   │   ├── `mask_dir`

    """

    def __init__(self, target_root_dir: str,
                 origin_data_dir: str,
                 transpose_order: Tuple,
                 leading_modality=None,
                 target_zero_ratio=0.9,
                 image_size=None,
                 leave_patient_name=True):

        self.target_root_dir = target_root_dir
        self.origin_data_dir = origin_data_dir
        self.transpose_order = transpose_order
        self.leading_modality = leading_modality
        self.target_zero_ratio = target_zero_ratio  # TODO: mask included
        self.image_size = image_size
        self.leave_patient_name = leave_patient_name

    @staticmethod
    def create_empty_dirs(parent_dir_path, dir_names):
        # creating utilized directories
        for s in dir_names:
            set_dir = os.path.join(parent_dir_path, s)
            Path(set_dir).mkdir(exist_ok=True)

    def create_train_val_test_sets(self,
                                   paths_from_local_dirs,
                                   train_size=0.75,
                                   n_patients=-1,
                                   validation_size=0.1):
        # creating target directory if already exists.
        Path(self.target_root_dir).mkdir(exist_ok=False)
        # creating inner directories
        # in each of the returned lists (train, test, val)
        # the order goes as follows: t1, t2, flair
        patients_names = os.listdir(self.origin_data_dir)
        self.create_empty_dirs(self.origin_data_dir, self.DIVISION_SETS)

        # loading the data
        modalities_filepaths = get_nii_filepaths(self.origin_data_dir,
                                                 paths_from_local_dirs,
                                                 n_patients)

        # splitting filenames into train and test sets
        n_samples = len(list(modalities_filepaths.values())[0])
        n_train_samples = int(train_size * n_samples)
        n_val_samples = int(validation_size * n_samples)

        if n_val_samples <= 0:
            logging.warning(
                f"Validation set would be empty so the train set gonna be reduced.\nInput train_size: {train_size} validation_size: {validation_size}")
            n_val_samples = 1
            n_train_samples -= 1

        # if shuffle:
        #     filepaths = list(zip(t1_filepaths, t2_filepaths, flair_filepaths))
        #     random.shuffle(filepaths)
        #     t1_filepaths, t2_filepaths, flair_filepaths = zip(*filepaths)

        for s in self.DIVISION_SETS:
            if s == "train":
                lower_bound = 0
                upper_bound = n_train_samples
            elif s == "test":
                lower_bound = n_val_samples + n_train_samples
                upper_bound = n_samples + 1
            else:
                lower_bound = n_train_samples
                upper_bound = n_val_samples + n_train_samples

            train_test_split_sets = {}
            for modality, filepaths in modalities_filepaths.items():
                # extracting current (train, test or val) set based on the range
                current_set_filepaths = filepaths[lower_bound:upper_bound]
                # assigning the set to the given modality
                train_test_split_sets[modality] = current_set_filepaths
                # FOR NOW parent=True as the solution TODO: reconsider
                # creating directories for each patient
                # self.create_empty_dirs(os.path.join(self.origin_data_dir, s),  get_youngest_dir(current_set_filepaths))

            self.create_set(train_test_split_sets, s)

        logging.log(logging.INFO, f"\nSUCCESS\nCreated train and test directories in {self.target_root_dir} "
                                  f"from {n_train_samples} train, {n_val_samples} validation and {n_samples - n_train_samples - n_val_samples} "
                                  f"test 3D MRI images")

    def save_slices(self, slices, patient_name, modality, main_dir, slice_min_index):
        for slice_index in range(len(slices)):
            # creating the directory and the file name based on the patient name and the slice
            slice_dirname = f"slice{slice_min_index + slice_index}"
            slice_path = os.path.join(main_dir, patient_name, slice_dirname)
            Path(slice_path).mkdir(exist_ok=True, parents=True)
            # saving the slice
            slice_path = os.path.join(slice_path, f"{modality}{self.SLICES_FILE_FORMAT}")
            np.save(slice_path, slices[slice_index])

    def create_set(self, modality_paths, set_type_name):
        if self.leading_modality is not None:
            if self.leading_modality in modality_paths.keys():
                logging.log(logging.INFO, f"Leading modality is {self.leading_modality}, "
                                          f"the images will be trimmed according to images in this modality")
        else:
            self.leading_modality = list(modality_paths.keys())[0]
            logging.log(logging.INFO,
                        f"Leading modality wasn't provided, taking the first one: {self.leading_modality}, "
                        f"the images will be trimmed according to images in this modality")
        main_dir = os.path.join(self.target_root_dir, set_type_name)
        n_samples = len(modality_paths[self.leading_modality])
        logging.log(logging.INFO, f"Creating {main_dir}, which will have data from {n_samples} patients")

        # in the `main_dir` create a directory for each patient and slice
        # with corresponding modalities such that the directory structre is
        # │ `mian_dir`
        # │   ├── patient_name_slice e.g. UCSF_001_slice58
        # │   │   ├── network input modalities e.g.
        # │   │   ├── t1.npy
        # │   │   ├── t2.npy
        # │   │   ├── flair.npy
        # │   │   ├── `mask_dir`

        logging.log(logging.INFO, f"Processing the `leading_modality` ({self.leading_modality})")
        # first iterating over the leading_modality to extract the same slices range
        utilized_slices_indicies = []

        for filepath in modality_paths[self.leading_modality]:
            patient_name = get_youngest_dir(filepath)
            logging.log(logging.INFO, f"File processed {filepath}\nPatient: {patient_name} in process ...\n")

            # TODO: include all with the tumor mask
            slices, slice_indices = load_nii_slices(filepath,
                                                    self.transpose_order,
                                                    self.image_size,
                                                    self.MIN_SLICE_INDEX,
                                                    self.MAX_SLICE_INDEX,
                                                    target_zero_ratio=self.target_zero_ratio)

            self.save_slices(slices, patient_name, self.leading_modality, main_dir, slice_indices[0])
            utilized_slices_indicies.append(slice_indices)

        # having the `utilized_slices_range`
        for index in range(n_samples):
            for modality in modality_paths.keys():
                patient_name = get_youngest_dir(modality_paths[modality][index])
                # the leading modality was already processed
                if modality == self.leading_modality:
                    continue

                slice_index_range = utilized_slices_indicies[index]
                filepath = modality_paths[modality][index]
                logging.log(logging.INFO, f"File processed {filepath}\nPatient: {patient_name} in process ...\n")
                slices, _ = load_nii_slices(filepath,
                                            self.transpose_order,
                                            self.image_size,
                                            min_slice_index=slice_index_range[0],
                                            max_slices_index=slice_index_range[-1],
                                            target_zero_ratio=self.target_zero_ratio)

                self.save_slices(slices, patient_name, modality, main_dir, slice_index_range[0])


def trim_image(image, target_image_size: Tuple[int, int]):
    x_pixels_margin = int((image.shape[0] - target_image_size[0]) / 2)
    y_pixels_margin = int((image.shape[1] - target_image_size[1]) / 2)

    if x_pixels_margin < 0 or y_pixels_margin < 0:
        raise ValueError(f"Target image size: {target_image_size} greater than original image size {image.shape}")

    return image[x_pixels_margin:target_image_size[0] + x_pixels_margin,
           y_pixels_margin:target_image_size[1] + y_pixels_margin]


def load_nii_slices(filepath: str, transpose_order, image_size: Optional[Tuple[int, int]] = None, min_slice_index=-1,
                    max_slices_index=-1, target_zero_ratio=0.9):
    def get_optimal_slice_range(brain_slices, target_zero_ratio=0.9):
        pixel_counts = np.unique(img, return_counts=True)

        # if there is less than 30% of the most frequent pixel there is a risk that the background is not unified
        if pixel_counts[1][0] / img.flatten().shape[0] < 0.3:
            logging.log(logging.WARNING, "The method assumes that all the background pixels have the same value. "
                                         f"In the provided volume {filepath} less than 30% of pixels have the same value.")
        background_color = pixel_counts[0][0]
        zero_ratios = np.array([np.sum(brain_slice == background_color) / (brain_slice.shape[0] * brain_slice.shape[1])
                                for brain_slice in brain_slices])
        satisfying_given_ratio = np.where(zero_ratios < target_zero_ratio)[0]

        return satisfying_given_ratio

    # noinspection PyUnresolvedReferences
    file_extension = os.path.splitext(filepath)[1]
    if file_extension == ".npy":
        img = np.load(filepath)
    elif file_extension == ".nii":
        img = nib.load(filepath).get_fdata()
    else:
        raise ValueError(f"Wrong file type provided in {filepath}, expected: .npy or .nii.gz")

    if max_slices_index > img.shape[-1]:  # img.shape[-1] == total number of slices
        raise ValueError("max_slices_index > img.shape[-1]")

    # in case of brain image being in wrong shape
    # we want (n_slice, img_H, img_W)
    # it changes from (img_H, img_W, n_slices) to desired length
    if transpose_order is not None:
        img = np.transpose(img, transpose_order)

    if image_size is not None:
        img = [trim_image(brain_slice, image_size) for brain_slice in img]

    if min_slice_index == -1 or max_slices_index == -1:
        taken_indices = get_optimal_slice_range(img, target_zero_ratio=target_zero_ratio)
        logging.log(logging.INFO, f"Slice range used for file {filepath}: <{min(taken_indices)}, {max(taken_indices)}>")
    else:
        logging.log(logging.INFO, f"Slice range used for file {filepath}: <{min_slice_index, max_slices_index}>")
        taken_indices = range(min_slice_index, max_slices_index)

    selected_slices = [img[slice_index] for slice_index in taken_indices]

    return selected_slices, taken_indices


def get_nii_filepaths(data_dir, filepaths_from_data_dir: Dict, n_patients=-1, shuffle_local_dirs=False):
    local_dirs = os.listdir(data_dir)

    if shuffle_local_dirs:
        random.shuffle(local_dirs)

    # if not specified taking all patients
    if n_patients == -1:
        n_patients = len(local_dirs)

    modalities_filepaths = {modality: [] for modality in filepaths_from_data_dir.keys()}

    i = 0
    for local_dir in local_dirs:
        if i >= n_patients:
            # loop runs until
            # all directories are visited (for ends)
            # the number of patients is fulfilled (i >= n_patients)
            break
        # just for one dataset purposes
        # inside_dir = local_dirs[i].split('_')[0]

        for modality, filepath_from_data_dir in filepaths_from_data_dir.items():
            alike_path = os.path.join(data_dir, local_dir, filepath_from_data_dir)
            retrieved_filepaths = sorted(glob(alike_path))

            if len(retrieved_filepaths) == 0:  # if not any found the directory is skipped
                break
            elif len(retrieved_filepaths) > 1:
                raise ValueError("More than one file with the provided regex: ", alike_path)

            modalities_filepaths[modality].extend(retrieved_filepaths)

        i += 1

    local_dirs_string = '\n'.join([loc_dir for loc_dir in local_dirs])
    modalities_counts = {modality: len(filepaths) for modality, filepaths in modalities_filepaths.items()}

    logging.log(logging.INFO,
                f"For the provided parameters, found {modalities_counts} by reading from files (with limit of {n_patients} patients):\n"
                f"{local_dirs_string}\n\n")

    return modalities_filepaths


# def try_create_dir(dir_name, allow_overwrite=True):
#     # TODO: simplify (maybe the function not needed with Path
#     try:
#         Path(dir_name).mkdir(parents=True, exist_ok=allow_overwrite)
#     except FileExistsError:
#         if allow_overwrite:
#             logging.warning(
#                 f"Directory {dir_name} already exists. You may overwrite your files or create some collisions!")
#         else:
#             raise FileExistsError(
#                 f"Directory {dir_name} already exists. If you want to overwrite it change allow_overwrite for True")
#
#     except FileNotFoundError:
#         ex = FileNotFoundError(
#             f"The path {dir_name} to directory willing to be created doesn't exist. You are in {os.getcwd()}.")
#
#         traceback.print_exception(FileNotFoundError, ex, ex.__traceback__)


def get_brains_slices_info(dir_name):
    filenames = os.listdir(dir_name)
    patient_slices = {}

    # the file is in the format e.g. patient-Brats18_TCIA10_420_1_t1-slice108.npy
    # we are extracting ID which is always between "-"
    # in this case Brats18_TCIA10_420_1_t1
    # list(set(...)) for extracting unique values
    patients_id = list(set([f.split('-')[1] for f in filenames]))
    # patients_id = list(set([f.split('-')[-2] for f in filenames]))

    for patient_id in patients_id:
        slices_nr = []
        for f in filenames:
            if patient_id in f:
                slices_nr.append(int(re.search(r'slice(\d+)', f).group(1)))

        patient_slices[patient_id] = (min(slices_nr), max(slices_nr))

    return patient_slices


# def get_all_brains_slices_info(ds_dir_name):
#     test_patients_slices = get_brains_slices_info(os.path.join(ds_dir_name, "test"))
#     train_patients_slices = get_brains_slices_info(os.path.join(ds_dir_name, "train"))
#     validation_patients_slices = get_brains_slices_info(os.path.join(ds_dir_name, "validation"))

#     return train_patients_slices, test_patients_slices, validation_patients_slices


def get_youngest_dir(filepath):
    return filepath.split(os.path.sep)[-2]


def test_mask_in(img_name, img_dir, breakpoint=10, failed_dir="failed"):
    img = np.load(os.path.join(img_dir, img_name))

    if np.sum(img[:, :breakpoint]):
        plt.imshow(img > 0)
        plt.savefig(os.path.join(failed_dir, img_name))
        print("\n\nWRONG MASKS IN THE IMAGE: ", img_name)

        return False

    else:
        return True
