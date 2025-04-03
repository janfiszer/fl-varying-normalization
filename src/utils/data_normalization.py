import logging
from typing import Dict, List
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os

from src.utils import files_operations as fop

import nibabel as nib
from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.fcm import FCMNormalize
from intensity_normalization.normalize.nyul import NyulNormalize
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.normalize.zscore import ZScoreNormalize


class NoNormalization:
    def __init__(self):
        return

    def __call__(self, image, *arg, **kwargs):
        return image


class MinMaxNormalization:
    def __init__(self, min_value=0, max_value=1):
        self.min_value = min_value
        self.max_value = max_value

        return

    def __call__(self, image, *args, **kwargs):
        values_range = self.max_value - self.min_value

        return (image - np.min(image)) / np.max(image) * values_range + self.min_value


class Normalizer:
    AVAILABLE_NORMALIZATION = ["nonorm", "nyul", "whitestripe", "fcm", "zscore", "minmax"]

    def __init__(self,
                 name: str,
                 normalizer,
                 normalizer_kwargs: Dict = None,
                 before_each_normalization_func: Callable[Dict] = None,
                 requirements=None):

        if normalizer_kwargs is None:
            normalizer_kwargs = {}

        if name not in self.AVAILABLE_NORMALIZATION:
            raise ValueError("Wrong `name` provided the only possible names are: ", *self.AVAILABLE_NORMALIZATION)

        self.name = name
        self.normalizer = normalizer(**normalizer_kwargs)
        self.requirements = requirements
        self.before_each_normalization_func = before_each_normalization_func

    def setup(self, *args):
        if self.name == "nyul":
            self.nyul_setup(*args)
        else:
            logging.log(logging.INFO, f"Nothing to set up in {self.name} normalizer")

    def __call__(self, image,
                 modality: str,
                 *args
                 ):

        kwargs = {"modality": modality}
        # so far skipping FCM and no playing with each iteration adaptation
        if self.name == "fcm":
            self.fcm_each_normalization(*args)

        return self.normalizer(image, **kwargs)

    def __str__(self):
        return self.name

    def nyul_setup(self, other_images):
        # loading the images
        self.normalizer.fit(other_images)

        return {}

    def fcm_each_normalization(self, t1_image):
        # TODO: double check if it should be done at each time
        _ = self.normalizer(t1_image)

        return {}


def plot_histogram(image, title=""):
    non_zeros = image[image > 0]
    plt.hist(non_zeros.flatten(), bins=100)
    plt.title(title)
    plt.xlabel("Pixel intensity")
    plt.show()


def normalize_fcm(image_to_normalize, t1_image, modality):
    fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
    _ = fcm_norm(t1_image)

    normalized_image = fcm_norm(image_to_normalize, modality=modality)

    return normalized_image


def normalize_nyul(image_to_normalize, modality, flair_images):
    nyul_norm = NyulNormalize(output_min_value=0.0, output_max_value=1.0)
    nyul_norm.fit(flair_images, modality=modality)
    normalized_image = nyul_norm(image_to_normalize, modality=modality)

    return normalized_image


def normalize_white_stripe(image_to_normalize, modality, norm_value=0.05):
    whitestripe_norm = WhiteStripeNormalize(norm_value=norm_value)
    normalized_image = whitestripe_norm(image_to_normalize, modality=modality)

    return normalized_image


def normalize_zscore(image_to_normalize, modality):
    zscore_norm = ZScoreNormalize()
    normalized_image = zscore_norm(image_to_normalize, modality=modality)

    return normalized_image


def generate_scans_paths(data_dir, suffix="flair"):
    # returns the relative path from the data_dir
    volumes_path = [os.path.join(data_dir, subject_dir, f"{subject_dir}_{suffix}.nii.gz") for subject_dir in
                    os.listdir(data_dir)]

    return volumes_path


def plot_single_histogram_and_slice(volume, slice_index, brain_mask, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Ensure the axes indexing works for multiple or single arrays
    ax_hist = axes[0]
    ax_image = axes[1]

    # Filter out the background
    positive_values = volume[brain_mask]

    # Plot histogram
    ax_hist.hist(positive_values, bins=100, color='blue', edgecolor='black')
    ax_hist.set_title(f'Histogram of pixel intensities values ({title}).')
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_ylabel('Frequency')

    # Extract a 2D slice (middle slice along the first axis)
    middle_slice = volume[:, :, slice_index]

    # Plot 2D slice as an image
    im = ax_image.imshow(middle_slice, cmap='grey')
    ax_image.set_title(f'Slice {slice_index}')
    plt.colorbar(im, ax=ax_image)

    plt.tight_layout()
    plt.savefig(filename)

    plt.close()


def plot_histograms_one_slice(volumes, slice_index, brain_mask, filename=None):
    """
    Takes a dict of 3D volumes s and plots:
    - Histogram of positive values
    - A 2D slice of the 3D array as an image with a colorbar
    All on a single subplot.
    """
    num_volumes = len(volumes)
    fig, axes = plt.subplots(num_volumes, 3, figsize=(10, 3 * num_volumes))

    for i, (normalization_name, volume) in enumerate(volumes.items()):
        # Ensure the axes indexing works for multiple or single arrays
        ax_hist = axes[i, 0]
        ax_image1 = axes[i, 1]
        ax_image2 = axes[i, 2]

        # Filter out the background
        positive_values = volume[brain_mask]

        # Plot histogram
        ax_hist.hist(positive_values, bins=100, color='blue', edgecolor='black')
        ax_hist.set_title(f'Normlization method: {normalization_name}')
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_ylabel('Frequency')

        # Extract a 2D slice (middle slice along the first axis)
        middle_slice1 = volume[:, :, slice_index[0]]
        middle_slice2 = volume[:, :, slice_index[1]]

        # Plot 2D slice as an image
        im = ax_image1.imshow(middle_slice1, cmap='grey')
        ax_image1.set_title(f'Slice {slice_index[0]}')
        plt.colorbar(im, ax=ax_image1)

        # Plot 2D slice as an image
        im = ax_image2.imshow(middle_slice2, cmap='grey')
        ax_image2.set_title(f'Slice {slice_index[1]}')
        plt.colorbar(im, ax=ax_image2)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()


def normalize_all_from_dir(data_dir: str,
                           output_dir: str,
                           path_from_local_dir: Dict,
                           normalizers: List[Normalizer],
                           not_normalize: List = None,
                           save_histogram_slice_plots=True, 
                           n_patients=-1):
    logging.log(logging.INFO, "Process of normalization and division af the dataset: STARTING...\n\n")

    if not_normalize is None:
        logging.log(logging.WARNING, "All the slices will be normalized. If for example there is a mask provide `not_normalize` to exclude it.\n")

    modalities_filepaths = fop.get_nii_filepaths(data_dir, path_from_local_dir, shuffle_local_dirs=True, n_patients=n_patients)

    # splitting the datasets into n subsets (n number of normalizers)
    n_normalization = len(normalizers)
    subset_size = len(list(modalities_filepaths.values())[0]) // n_normalization
    normalizers_with_indices_ranges = {normalizer: (i * subset_size, (i + 1) * subset_size) for i, normalizer in
                                       enumerate(normalizers)}

    logging.log(logging.INFO, f"{n_normalization} normalizers were provided, each of them will have a subset "
                              f"of {subset_size} patients and will be in aproriate directories in {output_dir}.\n")

    if save_histogram_slice_plots:
        histogram_slice_plot_dir = os.path.join(output_dir, "slices_and_histograms")
        Path(histogram_slice_plot_dir).mkdir(exist_ok=True, parents=True)

    # for each modality:
    # load the data and normalize the data with the given normalizer
    # store it in a new directory
    # loading t1 images if the normalizing method is WhiteStripe
    for normalizer, indices_range in normalizers_with_indices_ranges.items():
        logging.log(logging.INFO, f"Current normalizer: {normalizer}\nProcessing...")
        normalizer_path = os.path.join(output_dir, str(normalizer))

        Path(normalizer_path).mkdir(exist_ok=True)

        if normalizer.name == "fcm":
            # loading all the
            raw_t1_volumes = []
            for volume_path_index in range(indices_range[0], indices_range[1]):
                volume = nib.load(modalities_filepaths['t1'][volume_path_index]).get_fdata()
                raw_t1_volumes.append(volume)

        for modality, filepaths in modalities_filepaths.items():
            # dedicated_filepaths = filepaths[indices_range[0]: indices_range[1]]
            raw_volumes = []

            for volume_path_index in range(indices_range[0], indices_range[1]):
                volume = nib.load(filepaths[volume_path_index]).get_fdata()
                raw_volumes.append(volume)

            # so far I give as the argument of the function all the raw volumes always, because only one setup exists
            # TODO: make it more adaptive
            normalizer.setup([np.array(raw_volumes)])

            for i, volume in enumerate(raw_volumes):
                # extracting the name of the patient (directory the .nii.gz file is in)
                current_filepath = filepaths[indices_range[0] + i]
                patient_file_name = fop.get_youngest_dir(current_filepath)

                # creating a new diretory where all the images of the given patient will be saved
                patient_new_path = os.path.join(normalizer_path, patient_file_name)

                Path(patient_new_path).mkdir(exist_ok=True)

                # creating the filepath where the currently processed volume will be saved
                save_path = os.path.join(patient_new_path, f"{modality}.npy")

                # in case we have loaded the mask the normalization is not needed, so skip
                if modality not in not_normalize:
                    # list of arguments that function is using before normalization of each volume
                    each_normalization_args = []

                    # depending on the type of normalization different arguments used, so far only `FCM (Fuzzy c-means)`
                    if normalizer.name == "fcm":
                        each_normalization_args.append(raw_t1_volumes[i])

                    # actual normalization
                    normalized_volume = normalizer(volume, Modality.from_string(modality), *each_normalization_args)

                    logging_message = f"Volume from file '{current_filepath}' normalized by '{normalizer} normalizer' and saved to '{save_path}'."

                    # saving histogram and slice plot
                    if save_histogram_slice_plots:
                        image_path = os.path.join(histogram_slice_plot_dir,
                                                  f"{str(normalizer)}_{patient_file_name}_{modality}.png")
                        plot_single_histogram_and_slice(normalized_volume,
                                                        slice_index=110,
                                                        brain_mask=volume > 1e-6,
                                                        title=str(normalizer),
                                                        filename=image_path)

                else:
                    logging_message = f"Mask volume from file '{current_filepath}' saved to '{save_path}'."

                # saving the volume as a 3D numpy array
                logging.log(logging.INFO, logging_message)
                np.save(save_path, normalized_volume)

    logging.log(logging.INFO, "Process of normalization and division af the dataset: ENDED")


def test_every_normalizer():
    data_dir = "C:\\Users\\JanFiszer\\data\\mri\\flair_volumes"
    subject_1 = "Brats18_TCIA13_654_1"

    # loading t1w and flair images from the same subject
    image_t1 = nib.load(os.path.join(data_dir, subject_1,
                                     f"{subject_1}_t1.nii.gz")).get_fdata()  # assume skull-stripped otherwise load mask too
    image_flair = nib.load(os.path.join(data_dir, subject_1, f"{subject_1}_flair.nii.gz")).get_fdata()

    modality = Modality.FLAIR
    # subject_2 = "Brats18_TCIA13_654_1"

    # loading t1w and flair images from the same subject
    # image = nib.load(os.path.join(data_dir, subject_2,
    #                               f"{subject_2}_t1.nii.gz")).get_fdata()  # assume skull-stripped otherwise load mask too
    # image_flair = nib.load(os.path.join(data_dir, subject_2, f"{subject_2}_flair.nii.gz")).get_fdata()

    flair_volume_paths = generate_scans_paths(os.path.join(data_dir), suffix="flair")
    flair_images = [nib.load(volume_path).get_fdata() for volume_path in flair_volume_paths]
    brain_mask = image_t1 > 1e-6

    normalized_volumes = {"Not normalized": image_flair,
                          "FCM": normalize_fcm(image_flair, image_t1, modality),
                          "Nyul": normalize_nyul(image_flair, modality, flair_images),
                          # "White Stripe (normalization scaler=0.2)": normalize_white_stripe(image_flair, norm_value=0.2),
                          # "White Stripe (normalization scaler=0.02)": normalize_white_stripe(image_flair, norm_value=0.02),
                          "White Stripe (normalization scaler=0.05)": normalize_white_stripe(image_flair, modality,
                                                                                             norm_value=0.05),
                          "z-score": normalize_zscore(image_flair, modality=modality)
                          }

    plot_histograms_one_slice(normalized_volumes, slice_index=[110, 125], brain_mask=brain_mask)

    # plot histograms and example slice
    # plot_histogram(image_flair, title="Not normalized")
    # plot_histogram(nyul_flair, title="Nyul normalized")
    # plt.imshow(image_flair[:, :, slice_index], cmap="grey")
    # plt.colorbar()
    # plt.show()
    # plt.imshow(nyul_flair[:, :, slice_index], cmap="grey")
    # plt.colorbar()
    # plt.show()


def demonstrate_normalization(data_dir: str,
                              output_dir: str,
                              path_from_local_dir: Dict,
                              normalizers: List[Normalizer],
                              n_volumes: int):
    modalities_filepaths = fop.get_nii_filepaths(data_dir, path_from_local_dir, n_patients=n_volumes)

    Path(output_dir).mkdir(exist_ok=True)

    # loading t1 first since
    raw_t1_volumes = []
    for volume_path_index in modalities_filepaths['t1']:
        volume = nib.load(volume_path_index).get_fdata()
        raw_t1_volumes.append(volume)

    for modality, filepaths in modalities_filepaths.items():
        modality_dir = os.path.join(output_dir, modality)

        Path(modality_dir).mkdir(exist_ok=True)

        raw_volumes = []
        for volume_filepath in filepaths:
            volume = nib.load(volume_filepath).get_fdata()
            raw_volumes.append(volume)

        normalizer_and_volumes = {}

        for i, volume in enumerate(raw_volumes):
            for normalizer in normalizers:
                normalizer.setup([np.array(raw_volumes)])
                if normalizer.name == "fcm":
                    normalized_volume = normalizer(volume, Modality.from_string(modality), raw_t1_volumes[i])
                else:
                    normalized_volume = normalizer(volume, Modality.from_string(modality))

                normalizer_and_volumes[str(normalizer)] = normalized_volume

            patient_file_name = fop.get_youngest_dir(filepaths[i])

            plot_path = os.path.join(modality_dir, f"normalization_effect{patient_file_name}.png")
            logging.log(logging.INFO, f"Demo of normalization will be save to {plot_path}")
            plot_histograms_one_slice(normalizer_and_volumes, slice_index=[110, 125], brain_mask=volume > 0, filename=plot_path)


def define_normalizers_and_more():
    normalizers = [Normalizer("nonorm", NoNormalization),
                   Normalizer("nyul", NyulNormalize,
                              normalizer_kwargs={"output_min_value": -1.0, "output_max_value": 1.0}),
                   Normalizer("whitestripe", WhiteStripeNormalize, normalizer_kwargs={"norm_value": 0.05}),
                   Normalizer("zscore", ZScoreNormalize),
                   Normalizer("fcm", FCMNormalize),
                   Normalizer("minmax", MinMaxNormalization)
                   ]

    return normalizers
