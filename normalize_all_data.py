from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import os

from src.utils import files_operations as fop

import nibabel as nib
from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.fcm import FCMNormalize
from intensity_normalization.normalize.nyul import NyulNormalize
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.normalize.kde import KDENormalize

class Normalizer:
    def __init__(self, name, normalizing_function, requirements=None):
        self.name = name
        self.function = normalizing_function
        self.requirements = requirements

    def __call__(self, image):
        return self.function(image)

    def __str__(self):
        return self.name


def plot_histogram(image, title=""):
    non_zeros = image[image > 0]
    plt.hist(non_zeros.flatten(), bins=100)
    plt.title(title)
    plt.xlabel("Pixel intensity")
    plt.show()


def normalize_fcm(t1_image, image_flair=None):
    fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
    _ = fcm_norm(t1_image)

    if image_flair is None:
        image_flair = t1_image
    fcm_flair = fcm_norm(image_flair, modality=Modality.FLAIR)

    return fcm_flair


def normalize_nyul(flair_images, image_flair):
    nyul_norm = NyulNormalize(output_min_value=0.0, output_max_value=1.0)
    nyul_norm.fit(flair_images, modality=Modality.FLAIR)
    nyul_flair = nyul_norm(image_flair, modality=Modality.FLAIR)

    return nyul_flair


def normalize_white_stripe(image_flair, norm_value=1.0):
    whitestripe_norm = WhiteStripeNormalize(norm_value=norm_value)
    white_stripe_flair = whitestripe_norm(image_flair, modality=Modality.FLAIR)

    return white_stripe_flair


def generate_scans_paths(data_dir, suffix="flair"):
    # returns the relative path from the data_dir
    volumes_path = [os.path.join(data_dir, subject_dir, f"{subject_dir}_{suffix}.nii.gz") for subject_dir in os.listdir(data_dir)]

    return volumes_path


def plot_histograms_one_slice(volumes, slice_index, brain_mask):
    """
    Takes a dict of 3D volumes s and plots:
    - Histogram of positive values
    - A 2D slice of the 3D array as an image with a colorbar
    All on a single subplot.
    """
    num_volumes = len(volumes)
    fig, axes = plt.subplots(num_volumes, 2, figsize=(10, 5 * num_volumes))

    for i, (normalization_name, volume) in enumerate(volumes.items()):
        # Ensure the axes indexing works for multiple or single arrays
        ax_hist = axes[i, 0]
        ax_image = axes[i, 1]

        # Filter out the background
        positive_values = volume[brain_mask]

        # Plot histogram
        ax_hist.hist(positive_values, bins=100, color='blue', edgecolor='black')
        ax_hist.set_title(f'Histogram of pixel intensities values ({normalization_name})')
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_ylabel('Frequency')

        # Extract a 2D slice (middle slice along the first axis)
        middle_slice = volume[:, :, slice_index]

        # Plot 2D slice as an image
        im = ax_image.imshow(middle_slice, cmap='grey')
        ax_image.set_title(f'Slice {slice_index}')
        plt.colorbar(im, ax=ax_image)

    plt.tight_layout()
    plt.show()


def normalize_all_from_dir(data_dir, output_dir, path_from_local_dir: Dict, normalizers):
    modalities_filepaths = fop.get_nii_filepaths(data_dir, path_from_local_dir, shuffle_local_dirs=True)

    # splitting the datasets into n subsets (n number of normalizers)
    n_normalization = len(normalizers)
    subset_size = len(list(modalities_filepaths.values())[0]) // n_normalization
    normalizers_with_indices_ranges = {normalizer: (i*subset_size, (i+1)+subset_size) for i, normalizer in enumerate(normalizers)}

    # for each modality:
    # load the data and normalize the data with the given normalizer
    # store it in a new directory
    for modality, filepaths in modalities_filepaths.items():
        for normalizer, indices_range in normalizers_with_indices_ranges.items():
            dedicated_filepaths = filepaths[indices_range[0]: indices_range[1]]
            raw_volumes = []

            for volume_path in dedicated_filepaths:
               volume = nib.load(volume_path).get_fdata()
               raw_volumes.append(volume)

            for i, volume in enumerate(raw_volumes):
                normalized_volume = normalizer(volume)
                # normalized_volume = normalizer(volume, raw_volumes)

                # TODO: only substantial part of the brain
                # TODO: save as 3D volume?
                save_path = os.path.join(output_dir, f"{str(normalizer)}_{i}")
                np.save(save_path, normalized_volume)


def main():
    data_dir = "C:\\Users\\JanFiszer\\data\\mri\\flair_volumes"
    subject_1 = "Brats18_TCIA13_654_1"

    # loading t1w and flair images from the same subject
    image = nib.load(os.path.join(data_dir, subject_1,
                                  f"{subject_1}_t1.nii.gz")).get_fdata()  # assume skull-stripped otherwise load mask too
    image_flair = nib.load(os.path.join(data_dir, subject_1, f"{subject_1}_flair.nii.gz")).get_fdata()

    # subject_2 = "Brats18_TCIA13_654_1"

    # loading t1w and flair images from the same subject
    # image = nib.load(os.path.join(data_dir, subject_2,
    #                               f"{subject_2}_t1.nii.gz")).get_fdata()  # assume skull-stripped otherwise load mask too
    # image_flair = nib.load(os.path.join(data_dir, subject_2, f"{subject_2}_flair.nii.gz")).get_fdata()

    flair_volume_paths = generate_scans_paths(os.path.join(data_dir), suffix="flair")
    flair_images = [nib.load(volume_path).get_fdata() for volume_path in flair_volume_paths]
    brain_mask = image > 1e-6

    normalized_volumes = {"Not normalized": image_flair,
                          # "FCM": normalize_fcm(image, image_flair),
                          # "Nyul": normalize_nyul(flair_images, image_flair),
                          "White Stripe (normalization scaler=0.2)": normalize_white_stripe(image_flair, norm_value=0.2),
                          "White Stripe (normalization scaler=0.02)": normalize_white_stripe(image_flair, norm_value=0.02),
                          "White Stripe (normalization scaler=0.05)": normalize_white_stripe(image_flair, norm_value=0.05)}

    plot_histograms_one_slice(normalized_volumes, slice_index=110, brain_mask=brain_mask)

    # plot histograms and example slice
    # plot_histogram(image_flair, title="Not normalized")
    # plot_histogram(nyul_flair, title="Nyul normalized")
    # plt.imshow(image_flair[:, :, slice_index], cmap="grey")
    # plt.colorbar()
    # plt.show()
    # plt.imshow(nyul_flair[:, :, slice_index], cmap="grey")
    # plt.colorbar()
    # plt.show()


if __name__ == '__main__':
    paths_from_local_dirs = {"t1": "*t1.nii.gz", "t2": "*t2.nii.gz"}
    normalizers = [Normalizer("FCM", normalize_fcm), Normalizer("WhiteStripe", normalize_white_stripe)]
    normalize_all_from_dir("C:\\Users\\JanFiszer\\data\\mri\\flair_volumes", "C:\\Users\\JanFiszer\\data\\mri\\normalized",
                           paths_from_local_dirs, normalizers)

