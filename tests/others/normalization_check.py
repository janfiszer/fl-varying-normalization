import os

import nibabel as nib

from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.fcm import FCMNormalize
from intensity_normalization.normalize.nyul import NyulNormalize
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.normalize.zscore import ZScoreNormalize

from src.utils.data_normalization import plot_histograms_one_slice


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