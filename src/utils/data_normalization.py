import logging
from typing import Dict, List
from collections.abc import Callable

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


class Normalizer:
    def __init__(self,
                 name: str,
                 normalizer,
                 normalizer_kwargs: Dict = None,
                 before_each_normalization_func: Callable[Dict] = None,
                 requirements=None):

        if normalizer_kwargs is None:
            normalizer_kwargs = {}

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
                 # before_each_func_kwargs: Dict
                 ):

        kwargs = {"modality": modality}
        # so far skipping FCM and no playing with each iteration adaptation
        # for parameter, value in self.before_each_normalization_func(**before_each_func_kwargs):
        #     kwargs[parameter] = value

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
    volumes_path = [os.path.join(data_dir, subject_dir, f"{subject_dir}_{suffix}.nii.gz") for subject_dir in os.listdir(data_dir)]

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


def normalize_all_from_dir(data_dir: str,
                           output_dir: str,
                           path_from_local_dir: Dict,
                           normalizers: List[Normalizer],
                           save_histogram_slice_plots=True):
    modalities_filepaths = fop.get_nii_filepaths(data_dir, path_from_local_dir, shuffle_local_dirs=True, n_patients=12)

    # splitting the datasets into n subsets (n number of normalizers)
    n_normalization = len(normalizers)
    subset_size = len(list(modalities_filepaths.values())[0]) // n_normalization
    normalizers_with_indices_ranges = {normalizer: (i*subset_size, (i+1)*subset_size) for i, normalizer in enumerate(normalizers)}

    if save_histogram_slice_plots:
        histogram_slice_plot_dir = os.path.join(output_dir, "slices_and_histograms")
        fop.try_create_dir(histogram_slice_plot_dir)
    # for each modality:
    # load the data and normalize the data with the given normalizer
    # store it in a new directory
    for modality, filepaths in modalities_filepaths.items():
        # modality_path = os.path.join(output_dir, modality)
        # fop.try_create_dir(modality_path)
        for normalizer, indices_range in normalizers_with_indices_ranges.items():
            dedicated_filepaths = filepaths[indices_range[0]: indices_range[1]]
            raw_volumes = []

            for volume_path in dedicated_filepaths:
               volume = nib.load(volume_path).get_fdata()
               raw_volumes.append(volume)

            # so far I give as the argument of the function all the raw volumes always, because only one setup exists
            # TODO: make it more adaptive
            normalizer.setup([np.array(raw_volumes)])

            for i, volume in enumerate(raw_volumes):
                normalized_volume = normalizer(volume, Modality.from_string(modality))

                patient_file_name = dedicated_filepaths[i].split(os.path.sep)[-2]

                # saving histogram and slice plot
                if save_histogram_slice_plots:
                    image_path = os.path.join(histogram_slice_plot_dir, f"{str(normalizer)}_{patient_file_name}_{modality}.png")
                    plot_single_histogram_and_slice(normalized_volume,
                                                    slice_index=110,
                                                    brain_mask=volume > 1e-6,
                                                    title=str(normalizer),
                                                    filename=image_path)

                # saving the volume as a 3D numpy array
                patient_new_path = os.path.join(output_dir, patient_file_name)
                fop.try_create_dir(patient_new_path)
                save_path = os.path.join(patient_new_path, f"{str(normalizer)}_{modality}.npy")
                np.save(save_path, normalized_volume)


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
                          "White Stripe (normalization scaler=0.05)": normalize_white_stripe(image_flair, modality, norm_value=0.05),
                          "z-score": normalize_zscore(image_flair, modality=modality)
                          }

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


def define_normalizers_and_more():
    normalizers = [Normalizer("nonorm", NoNormalization),
                   Normalizer("nyul", NyulNormalize, normalizer_kwargs={"output_min_value": -1.0, "output_max_value": 1.0}),
                   Normalizer("whitestripe", WhiteStripeNormalize, normalizer_kwargs={"norm_value": 0.05}),
                   Normalizer("ZScore", ZScoreNormalize)
                   ]

    return normalizers
