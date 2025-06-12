import numpy as np
import os
from pycimg import CImg

data_dir = "C:\\Users\\JanFiszer\\data\\mri\\fl-varying-norm\\from-nonorm-normalized-UCSF-PDGM\\nonorm\\"
subject_dir = "UCSF-PDGM-0229_nifti"
subject_dir_path = os.path.join(data_dir, subject_dir)

t1_volume = np.load(os.path.join(subject_dir_path, "t1.npy"))
t2_volume = np.load(os.path.join(subject_dir_path, "t2.npy"))
flair_volume = np.load(os.path.join(subject_dir_path, "flair.npy"))
mask_volume = np.load(os.path.join(subject_dir_path, "mask.npy"))
transpose_order = (2,0,1)

mask_volume_scaler = np.max(t1_volume)

to_visualize = [t1_volume, t2_volume, flair_volume, (mask_volume > 0) * mask_volume_scaler]

all_transposed = [np.transpose(v, transpose_order) for v in to_visualize]
all_stacked = np.concatenate(all_transposed, axis=2)
CImg(all_stacked).display()
