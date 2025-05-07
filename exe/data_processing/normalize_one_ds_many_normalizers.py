import logging
import sys
from configs import config
from src.utils import data_normalization

if __name__ == '__main__':
    if config.LOCAL:
        n_patients = -1
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\nomralized-UCSF-PDGM\\nonorm"
        output_dir = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM"

        paths_from_local_dirs = {"t1": "*t1.nii.gz",
                                 "t2": "*t2.nii.gz",
                                 "flair": "*flair.nii.gz",
                                 "mask": "*tumor_segmentation.nii.gz"
                                 }
    else:
        n_patients = int(sys.argv[1])

        data_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-PDGM-v3"
        output_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-normalized"

    normalizers = data_normalization.define_normalizers_and_more()

    data_normalization.normalize_all_from_dir(data_dir,
                                                     output_dir,
                                                     config.MODALITIES_AND_NPY_PATHS_FROM_LOCAL_DIR,
                                                     normalizers,
                                                     not_normalize=["mask"],
                                                     n_patients=n_patients,
                                                     divide_dataset=False
                                                     )