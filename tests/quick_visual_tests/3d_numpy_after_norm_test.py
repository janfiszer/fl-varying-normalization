import os

import numpy as np
from matplotlib import pyplot as plt


# sample_patient_path = "C:\\Users\\JanFiszer\\data\\mri\\nomralized-UCSF-PDGM\\minmax\\UCSF-PDGM-0429_FU003d_nifti"
# sample_patient_path = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM\\fcm\\UCSF-PDGM-0491_nifti"
data_path = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM"
# sample_patient_path = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM\\nonorm"
# sample_patient_path = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM\\minmax"
# sample_patient_path = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM\\minmax"

slice_index = 111

for ds_name in ["fcm", "minmax", "nonorm", "nyul", "whitestripe", "zscore"]:
    sample_patient_path = os.path.join(data_path, ds_name, "UCSF-PDGM-0491_nifti")
    for volume_filename in os.listdir(sample_patient_path):
        volume_path = os.path.join(sample_patient_path, volume_filename)
        volume = np.load(volume_path)
        plt.imshow(volume[:, :, slice_index], cmap="gray")
        plt.title(f"{ds_name} {volume_filename}")
        plt.colorbar()
        plt.show()
