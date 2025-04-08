import os

import numpy as np
from matplotlib import pyplot as plt


sample_patient_path = "C:\\Users\\JanFiszer\\data\\mri\\nomralized-UCSF-PDGM\\minmax\\UCSF-PDGM-0429_FU003d_nifti"
slice_index = 111
for volume_filename in os.listdir(sample_patient_path):
    volume_path = os.path.join(sample_patient_path, volume_filename)
    volume = np.load(volume_path)
    plt.imshow(volume[:, :, slice_index], cmap="grey")
    plt.title(volume_filename)
    plt.colorbar()
    plt.show()
