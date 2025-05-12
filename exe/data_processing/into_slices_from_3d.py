import sys
import os
from src.utils.files_operations import TransformVolumesToNumpySlices, get_patients_filepaths
from configs import config


if __name__ == '__main__':
    if config.LOCAL:
        target_root_dir = "C:\\Users\\JanFiszer\\data\\mri\\minmax-from-nonorm"
        current_data_dir = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM\\minmax"
        n_patients = -1
    else:
        target_root_dir = sys.argv[1]
        current_data_dir = sys.argv[2]
        n_patients = int(sys.argv[3])

    paths_from_local_dirs = config.MODALITIES_AND_NPY_PATHS_FROM_LOCAL_DIR

    transformer = TransformVolumesToNumpySlices(target_root_dir,
                                                current_data_dir,
                                                mask_volume_filename=config.MASK_DIR,
                                                transpose_order=(2, 0, 1),
                                                target_zero_ratio=0.8,
                                                max_zero_ratio_on_slice_with_tumor=0.95) # not relevant for this action

    modality_paths = get_patients_filepaths(current_data_dir, paths_from_local_dirs, n_patients)

    dataset_name = os.path.basename(current_data_dir)

    transformer.create_set(modality_paths, dataset_name)
