import logging
import configs.config
from src.utils import data_normalization

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # test_every_normalizer()
    if configs.config.LOCAL:
        # data_dir = "C:\\Users\\JanFiszer\\data\\mri\\flair_volumes"
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\UCSF-PDGM"
        output_dir = "C:\\Users\\JanFiszer\\data\\mri\\nomralized-UCSF-PDGM"
        # output_dir = "C:\\Users\\JanFiszer\\data\\mri\\normalized"
        paths_from_local_dirs = {"t1": "*t1.nii.gz",
                                 "t2": "*t2.nii.gz",
                                 "flair": "*flair.nii.gz",
                                 # "mask": "*tumor_segmentation.nii.gz"
                                 }
    else:
        data_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-PDGM-v3"
        output_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-normalized"
        paths_from_local_dirs = {"t1": "*T1.nii.gz",
                                 "t2": "*T2.nii.gz",
                                 "flair": "*FLAIR.nii.gz",
                                 # "mask": "*tumor_segmentation.nii.gz"
                                 }
    normalizers = data_normalization.define_normalizers_and_more()

    data_normalization.demonstrate_normalization(data_dir,
                                                 output_dir,
                                                 paths_from_local_dirs,
                                                 normalizers,
                                                 n_volumes=len(normalizers))
    #
    # data_normalization.normalize_all_from_dir(data_dir,
    #                                           output_dir,
    #                                           paths_from_local_dirs,
    #                                           normalizers,
    #                                           mask_file="mask",
    #                                           )
