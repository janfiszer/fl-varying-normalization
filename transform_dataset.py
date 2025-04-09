import sys
import logging
from src.utils.files_operations import TransformVolumesToNumpySlices
from configs import config

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    if config.LOCAL:
        target_root_dir = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test"
        current_data_dir = "C:\\Users\\JanFiszer\\data\\mri\\nomralized-UCSF-PDGM\\whitestripe"
        n_patients = -1
    else:
        target_root_dir = sys.argv[1]
        current_data_dir = sys.argv[2]
        n_patients = int(sys.argv[3])


    paths_from_local_dirs = {"t1": "t1.npy",
                             "t2": "t2.npy",
                             "flair": "flair.npy",
                             "mask": "mask.npy"
                             }
    transformer = TransformVolumesToNumpySlices(target_root_dir,
                                                current_data_dir,
                                                mask_volume_filename="mask",
                                                transpose_order=(2, 0, 1),
                                                target_zero_ratio=0.8)

    transformer.create_train_val_test_sets(paths_from_local_dirs,
                                           train_size=0.75,
                                           validation_size=0.05,
                                           n_patients=n_patients)
