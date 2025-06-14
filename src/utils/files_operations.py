import logging
import os
import random
from glob import glob
from typing import Tuple, Optional, Dict, List, Set

from pathlib import Path
import numpy as np


class TransformVolumesToNumpySlices:
    # by investigation in eda.ipynb obtained
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
                 mask_volume_filename: str = None,
                 max_zero_ratio_on_slice_with_tumor: float = 0.95,
                 leading_modality=None,
                 target_zero_ratio=0.9,
                 image_size: Optional[Tuple[int, int]] = None,
                 leave_patient_name=True):

        self.target_root_dir = target_root_dir
        self.origin_data_dir = origin_data_dir
        self.transpose_order = transpose_order
        self.leading_modality = leading_modality
        self.target_zero_ratio = target_zero_ratio
        self.image_size = image_size
        self.max_zero_ratio_on_slice_with_tumor = max_zero_ratio_on_slice_with_tumor
        self.leave_patient_name = leave_patient_name
        self.mask_volume_name = mask_volume_filename

    @staticmethod
    def create_empty_dirs(parent_dir_path, dir_names):
        # creating utilized directories
        for s in dir_names:
            set_dir = os.path.join(parent_dir_path, s)
            Path(set_dir).mkdir(exist_ok=True)

    def create_train_val_test_sets(self,
                                   paths_from_local_dirs,
                                   n_patients=-1,
                                   train_size=0.75,
                                   validation_size=0.1):
        """
        Main method - creates the train, test and validations sets with the appropriate split.
        :param paths_from_local_dirs:  Dict in the form that `get_patients_filepaths()` returns.
        It has the paths (regex) from the inner directories for each modality
        e.g. { "t1": "*T1.nii.gz", "t2": "*T2.nii.gz", "flair": "*FLAIR.nii.gz", "mask": "*tumor_segmentation.nii.gz"}
        :param train_size: 0-1 value representing the percentage of the train data samples (test_size = m_samples - (train_size+validation_size)
        :param validation_size: 0-1 value representing the percentage of the validation data samples (test_size = m_samples - (train_size+validation_size)
        :param n_patients: Upper limit of the utilized patients (in case more than we need it the dataset)
        :return:
        """
        # creating target directory if already exists.
        Path(self.target_root_dir).mkdir(exist_ok=False)
        # creating inner directories
        # in each of the returned lists (train, test, val)
        self.create_empty_dirs(self.target_root_dir, self.DIVISION_SETS)

        # loading the data
        modalities_filepaths = get_patients_filepaths(self.origin_data_dir,
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

            self.create_set(train_test_split_sets, s)

        logging.log(logging.INFO, f"SUCCESS\nCreated train and test directories in {self.target_root_dir} "
                                  f"from {n_train_samples} train, {n_val_samples} validation and {n_samples - n_train_samples - n_val_samples} "
                                  f"test 3D MRI images")

    def save_slices(self, slices, patient_name, modality, main_dir, slice_min_index):
        # saves all the provided slices with the appropriate name
        for slice_index in range(len(slices)):
            # creating the directory and the file name based on the patient name and the slice
            slice_dirname = f"slice{slice_min_index + slice_index}"
            slice_path = os.path.join(main_dir, patient_name, slice_dirname)
            Path(slice_path).mkdir(exist_ok=True, parents=True)
            # saving the slice
            slice_path = os.path.join(slice_path, f"{modality}{self.SLICES_FILE_FORMAT}")
            np.save(slice_path, slices[slice_index])

    def create_set(self, modality_paths: Dict[str, List[str]], set_type_name):
        """
        Creates a set where each patient has it own directory in the `self.main_dir` create a directory for each patient and slice
        with corresponding modalities such that the directory structure is
        │ `mian_dir`
        │   ├── patient_name_slice e.g. UCSF_001_slice58
        │   │   ├── network input modalities e.g.
        │   │   ├── t1.npy
        │   │   ├── t2.npy
        │   │   ├── flair.npy
        │   │   ├── `mask_dir`

        :param modality_paths: Dict in the form that `get_patients_filepaths()` returns.
        It has the paths (regex) from the inner directories for each modality
        e.g. { "t1": "*T1.nii.gz", "t2": "*T2.nii.gz", "flair": "*FLAIR.nii.gz", "mask": "*tumor_segmentation.nii.gz"}
        :param set_type_name: The set directory name.
        """

        # verifying if all the provided parameters are valid
        if self.leading_modality is not None:
            if self.leading_modality in modality_paths.keys():
                logging.log(logging.INFO, f"Leading modality is {self.leading_modality}, "
                                          f"the images will be trimmed according to images in this modality")
        else:
            self.leading_modality = list(modality_paths.keys())[0]
            logging.log(logging.INFO,
                        f"Leading modality wasn't provided, taking the first one: {self.leading_modality}, "
                        f"the images will be trimmed according to images in this modality")

        if self.mask_volume_name:
            if self.mask_volume_name not in modality_paths.keys():
                raise ValueError(
                    f"Provided `paths_from_local_dirs` with keys: {modality_paths.keys()} do not include the"
                    f"`mask_volume_name` ({self.mask_volume_name})")

        # init the directory where the slices will be saved - `main_dir`
        main_dir = os.path.join(self.target_root_dir, set_type_name)
        n_samples = len(modality_paths[self.leading_modality])

        logging.log(logging.INFO, f"Creating {main_dir}, which will have data from {n_samples} patients")

        utilized_slices_indices: List[Set] = []

        # if it was provided get the slices were the mask is
        if self.mask_volume_name:
            logging.log(logging.INFO, f"`PROVIDED: mask_volume_name={self.mask_volume_name}`\n"
                                      f"Taking all the slices from this key in the provided `modality_paths`")
            for filepath_index in range(n_samples):
                # loading the currently processed volume from the `self.leading_modality`
                volume_filepath = modality_paths[self.leading_modality][filepath_index]
                volume = self.load_slice(volume_filepath)
                # finding the slices with zero ratio (percentage) below `self.max_zero_ratio_on_slice_with_tumor`
                slice_indices_with_minimum_brain_for_mask = self.get_optimal_slice_range(volume,
                                                                                         self.max_zero_ratio_on_slice_with_tumor,
                                                                                         slices_id=volume_filepath)

                # loading the mask volume
                mask_volume_filepath = modality_paths[self.mask_volume_name][filepath_index]
                mask_volume = self.load_slice(mask_volume_filepath)
                # finding the slices with mask volume
                slice_indices_with_mask = self.get_indices_mask_slices(mask_volume)

                # taking intersection of the two sets
                slices_indices_intersection = slice_indices_with_minimum_brain_for_mask.intersection(slice_indices_with_mask)
                utilized_slices_indices.append(slices_indices_intersection)
                logging.log(logging.INFO, f"In the directory {get_youngest_dir(mask_volume_filepath)} "
                                          f"{len(slices_indices_intersection)} slices with mask they were found")

            num_all_slices_with_mask = sum([len(i_slices) for i_slices in utilized_slices_indices])
            logging.log(logging.INFO,
                        f"Checking the slices with mask: COMPLETED\nIn total there are {num_all_slices_with_mask} slices with mask")

        # iterating over the `leading_modality` to extract the same slices range
        logging.log(logging.INFO, f"Processing the `leading_modality` ({self.leading_modality})")
        for i, filepath in enumerate(modality_paths[self.leading_modality]):
            patient_name = get_youngest_dir(filepath)
            logging.log(logging.DEBUG, f"File processed {filepath}\nPatient: {patient_name} in process...")
            current_utilized_slices = utilized_slices_indices[i]

            slices, slice_indices = self.smart_load_slices(filepath,
                                                           min_slices_index=min(current_utilized_slices),
                                                           max_slices_index=max(current_utilized_slices))

            self.save_slices(slices, patient_name, self.leading_modality, main_dir,
                             min(slice_indices.union(current_utilized_slices)))
            utilized_slices_indices[i].update(slice_indices)

        logging.info(f"The slices selection: COMPLETED\n"
                                  f"In total there are {sum([len(i_slices) for i_slices in utilized_slices_indices])} slices")

        # processing remaining modalities
        remaining_modalities = list(modality_paths.keys())
        remaining_modalities.remove(self.leading_modality)
        logging.info(f"Processing all the other modalities: {remaining_modalities}")

        # having the `utilized_slices_range`
        for index in range(n_samples):
            for modality in remaining_modalities:
                patient_name = get_youngest_dir(modality_paths[modality][index])

                slice_index_range = utilized_slices_indices[index]
                min_slices_index, max_slices_index = min(slice_index_range), max(slice_index_range)
                filepath = modality_paths[modality][index]
                logging.log(logging.DEBUG, f"File processed {filepath}\nPatient: {patient_name} in process...")
                slices, _ = self.smart_load_slices(filepath,
                                                   min_slices_index=min_slices_index,
                                                   max_slices_index=max_slices_index,
                                                   compute_optimal_slice_range=False)

                self.save_slices(slices, patient_name, modality, main_dir, min_slices_index)

    def smart_load_slices(self,
                          filepath: str,
                          min_slices_index=-1,
                          max_slices_index=-1,
                          compute_optimal_slice_range=True):
        img = self.load_slice(filepath)

        if max_slices_index > img.shape[-1]:  # img.shape[-1] == total number of slices
            raise ValueError("max_slices_index > img.shape[-1]")

        # in case of brain image being in wrong shape
        # we want (n_slice, img_H, img_W)
        # it changes from (img_H, img_W, n_slices) to desired length
        if self.image_size is not None:
            img = [trim_image(brain_slice, self.image_size) for brain_slice in img]

        # by default the range is given in the function
        upper_bounder = max_slices_index
        lower_bounder = min_slices_index

        if compute_optimal_slice_range:
            taken_indices = self.get_optimal_slice_range(img, self.target_zero_ratio, slices_id=filepath)
            if len(taken_indices) == 0:
                raise ValueError(
                    "No brain slices with the substantial portion of brain (greater than `target_zero_ratio`), increase this value.")
            min_substantial_brain = min(taken_indices)
            max_substantial_brain = max(taken_indices)

            if min_slices_index == -1 or min_substantial_brain < min_slices_index:
                lower_bounder = min_substantial_brain

            if max_slices_index == -1 or max_substantial_brain > max_slices_index:
                upper_bounder = max_substantial_brain
        else:
            if min_slices_index == -1 or max_slices_index == -1:
                raise ValueError("The `min_slices_index` and `max_slices_index` have to "
                                 "be provided in case the `compute_optimal_slice_range` is False")

        logging.log(logging.INFO, f"Slice range used for file {filepath}: <{lower_bounder}, {upper_bounder}>")

        taken_indices = range(lower_bounder, upper_bounder + 1)
        selected_slices = [img[slice_index] for slice_index in taken_indices]

        return selected_slices, set(taken_indices)

    def load_slice(self, filepath):
        # loading the file
        file_extension = os.path.splitext(filepath)[1]
        if file_extension == ".npy":
            img = np.load(filepath)
        else:
            raise ValueError(f"Wrong file type provided in {filepath}, expected: .npy")

        # in case of brain image being in wrong shape
        # we want (n_slice, img_H, img_W)
        # it changes from (img_H, img_W, n_slices) to desired length
        if self.transpose_order is not None:
            img = np.transpose(img, self.transpose_order)

        else:
            logging.warning(f"self.transpose_order not provided, usually the MRI volumes are in a flipped form. "
                            f"Desired is: (n_slice, img_H, img_W)")
        return img

    @staticmethod
    def get_optimal_slice_range(brain_slices, target_zero_ratio, slices_id=None):
        background_color = extract_background_pixel_value(brain_slices, slices_id)
        zero_ratios = np.array([np.sum(brain_slice == background_color) / (brain_slice.shape[0] * brain_slice.shape[1])
                                for brain_slice in brain_slices])
        satisfying_given_ratio = np.where(zero_ratios < target_zero_ratio)[0]

        return set(satisfying_given_ratio)

    @staticmethod
    def get_indices_mask_slices(mask_volume: np.ndarray):
        # retrieving slices where the slice have at least one pixel different from zero
        having_any_mask = {index for index, mask_slice in enumerate(mask_volume) if np.sum(mask_slice) > 0}
        return having_any_mask


def extract_background_pixel_value(image, image_name=None):
    if image_name is None:
        slices_id = ""
    pixel_counts = np.unique(image, return_counts=True)

    # if there is less than 30% of the most frequent pixel there is a risk that the background is not unified
    if pixel_counts[1][0] / image.flatten().shape[0] < 0.3:
        logging.log(logging.WARNING, "The method assumes that all the background pixels have the same value. "
                                     f"In the provided volume {image_name} less than 30% of pixels have the same value.")
    return pixel_counts[0][0]


def trim_image(image, target_image_size: Tuple[int, int]):
    x_pixels_margin = int((image.shape[0] - target_image_size[0]) / 2)
    y_pixels_margin = int((image.shape[1] - target_image_size[1]) / 2)

    if x_pixels_margin < 0 or y_pixels_margin < 0:
        raise ValueError(f"Target image size: {target_image_size} greater than original image size {image.shape}")

    return image[x_pixels_margin:target_image_size[0] + x_pixels_margin,
           y_pixels_margin:target_image_size[1] + y_pixels_margin]


def get_patients_filepaths(data_dir: str, filepaths_from_data_dir: Dict, n_patients=-1, shuffle_local_dirs=False) -> Dict[str, List[str]]:
    """
    Iterates over the inner directories and returns all the files (filepaths) matching the filepath from the data_dir
    :param data_dir: The directory where the data is stored
    :param filepaths_from_data_dir: Dict of the paths (regex) from the inner directories for each modality
    e.g. { "t1": "*T1.nii.gz", "t2": "*T2.nii.gz", "flair": "*FLAIR.nii.gz", "mask": "*tumor_segmentation.nii.gz"}
    :param n_patients: Upper limit of the utilized patients (in case more than we need it the dataset)
    :param shuffle_local_dirs: boolean for the value
    :return: Directory with keys of `filepaths_from_data_dir.keys()` and found list of the found filepaths
    """
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

        for modality, filepath_from_data_dir in filepaths_from_data_dir.items():
            alike_path = os.path.join(data_dir, local_dir, filepath_from_data_dir)
            retrieved_filepaths = sorted(glob(alike_path))

            if len(retrieved_filepaths) == 0:  # if not any found the directory is skipped
                logging.log(logging.WARNING,
                            f"In file {local_dir} there were no alike filepaths ({alike_path}), so the patient is skipped.")
                # removing last elements for this patient from the already processed modalities
                for modality_to_remove in filepaths_from_data_dir.keys():
                    # only until the current patients are removed
                    if modality_to_remove == modality:
                        break
                    modalities_filepaths[modality_to_remove] = modalities_filepaths[modality_to_remove][:-1]
                break
            elif len(retrieved_filepaths) > 1:
                raise ValueError("More than one file with the provided regex: ", alike_path)

            modalities_filepaths[modality].append(retrieved_filepaths[0])

        i += 1

    used_local_dirs_string = '\n'.join([loc_dir for loc_dir in local_dirs[:i]])
    modalities_counts = {modality: len(filepaths) for modality, filepaths in modalities_filepaths.items()}

    logging.log(logging.INFO,
                f"For the provided parameters, found {modalities_counts} by reading from files (with limit of {n_patients} patients):\n"
                f"{used_local_dirs_string}\n\n")

    return modalities_filepaths


def sort_by_substring_order(main_list, order_list):
    # Create a key function to determine the sort order
    def sort_key(string):
        for index, substring in enumerate(order_list):
            if substring in string:
                return index
        return len(order_list)  # Put items without a match at the end

    # Sort the main list using the key function
    sorted_list = sorted(main_list, key=sort_key)

    print(f"Sorted list: {sorted_list}, before {main_list}")
    return sorted_list


def filter_filepaths(modalities_filepaths: Dict[str, List[str]] = None, filtered_patients: List[str] = None):
    """
    Filters out the filepaths based on the patients names. If the filepath contains any of
    the provided patients names (substring) it is left, otherwise it's filtered out.
    :param modalities_filepaths: In the form that `get_patients_filepaths()` returns
    :param filtered_patients: Filter and also in the form that `get_patients_filepaths()` returns
    :return: Filtered filepaths.
    """
    filtered_filepaths_dict = {}

    for modality, filepaths in modalities_filepaths.items():
        filtered_filepaths = []
        
        for filepath in filepaths:
            # Check if any substring is in the current string
            if any(patient_name in filepath for patient_name in filtered_patients):
                filtered_filepaths.append(filepath)

        filtered_filepaths_dict[modality] = filtered_filepaths

    return filtered_filepaths_dict


def get_youngest_dir(filepath):
    filepath_split = filepath.split(os.path.sep)
    if len(filepath_split) < 2:
        raise ValueError(f"The provided filepath ({filepath}) is only a file name (no path separators)")
    return filepath_split[-2]
