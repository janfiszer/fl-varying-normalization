import logging
import sys
from configs import config
from src.utils import data_normalization, files_operations as fop

if __name__ == '__main__':
    if config.LOCAL:
        n_patients = -1
        # data_dir = "C:\\Users\\JanFiszer\\data\\mri\\flair_volumes"
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\UCSF-PDGM"
        output_dir = "C:\\Users\\JanFiszer\\data\\mri\\nomralized-UCSF-PDGM"
        # output_dir = "C:\\Users\\JanFiszer\\data\\mri\\normalized"
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

    modalities_filepaths = fop.get_patients_filepaths(data_dir,
                                                      config.MODALITIES_AND_NII_PATHS_FROM_LOCAL_DIR,
                                                      shuffle_local_dirs=True,
                                                      n_patients=n_patients)

    data_normalization.normalize_all_from_dir(modalities_filepaths,
                                              output_dir,
                                              normalizers,
                                              not_normalize=["mask"],
                                              divide_dataset=True
                                              )
