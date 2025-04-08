from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.fcm import FCMNormalize
from intensity_normalization.normalize.nyul import NyulNormalize
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize

import nibabel as nib
import os
import matplotlib.pyplot as plt

def plot_histogram(image, filename, title=""):
    non_zeros = image[image != 0]
    plt.hist(non_zeros.flatten(), bins=100)
    plt.title(title)
    plt.xlabel("Pixel intensity")
    plt.savefig(filename)

data_dir = "C:\\Users\\JanFiszer\\data\\mri\\flair_volumes"
subject_1 = "Brats18_TCIA13_653_1"


image = nib.load(os.path.join(data_dir, subject_1, f"{subject_1}_t1.nii.gz")).get_fdata()  # assume skull-stripped otherwise load mask too
fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
normalized_fcm = fcm_norm(image)

slice_index = 110

plt.imshow(image[:, :, slice_index], cmap="grey")
plt.colorbar()
plt.savefig("slice-not-norm.png")

plt.imshow(normalized_fcm[:, :, slice_index], cmap="grey")
plt.colorbar()
plt.savefig("slice-norm.png")

# image = nib.load(os.path.join(data_dir, subject_1, f"{subject_1}_t1.nii.gz")).get_fdata()  # assume skull-stripped otherwise load mask too

# fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
# normalized_fcm = fcm_norm(image)
# whitestripe_norm = WhiteStripeNormalize(norm_value=1.0)
# normalized_wm = whitestripe_norm(image, modality=Modality.T1)

# plot_histogram(image, "not-normalized.png", "Not normalized T1w image")
# plot_histogram(normalized_fcm, "normalized-wm.png", "FCM normalized T1w image")

print("check:", os.getcwd())

