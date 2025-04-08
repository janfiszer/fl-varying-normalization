import os

import numpy as np
from matplotlib import pyplot as plt

data_dir = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\train\\UCSF-PDGM-0431_FU001d_nifti\\slice128"

for filename in os.listdir(data_dir):
    print(os.path.join(data_dir, filename))
    slice_2d = np.load(os.path.join(data_dir, filename))
    plt.imshow(slice_2d, cmap="grey")
    plt.title(filename)
    plt.show()
