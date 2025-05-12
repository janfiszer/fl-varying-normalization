import logging
import sys
from configs import config
from src.utils import data_normalization, files_operations as fop

if __name__ == '__main__':
    # setting the paths
    if config.LOCAL:
        n_patients = -1
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\nomralized-UCSF-PDGM\\nonorm"
        output_dir = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM"
    else:
        n_patients = int(sys.argv[1])

        data_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-normalized/nonorm"
        output_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-1ds-normalized-test"

    # initializing the normalizers
    normalizers = data_normalization.define_normalizers_and_more()

    # getting the path from local_dir for all modalities excluding mask
    paths_from_local_dirs = config.MODALITIES_AND_NPY_PATHS_FROM_LOCAL_DIR
    paths_from_local_dirs.pop(config.MASK_DIR)

    # list of patients from no norm test set (this only will be processed)
    nonorm_test_patients = ["UCSF-PDGM-0516_nifti",
                            "UCSF-PDGM-0116_nifti",
                            "UCSF-PDGM-0179_nifti",
                            "UCSF-PDGM-0158_nifti",
                            "UCSF-PDGM-0345_nifti",
                            "UCSF-PDGM-0229_nifti",
                            "UCSF-PDGM-0440_nifti",
                            "UCSF-PDGM-0369_nifti",
                            "UCSF-PDGM-0225_nifti",
                            "UCSF-PDGM-0402_nifti",
                            "UCSF-PDGM-0349_nifti",
                            "UCSF-PDGM-0039_nifti",
                            "UCSF-PDGM-0486_nifti",
                            "UCSF-PDGM-0185_nifti",
                            "UCSF-PDGM-0330_nifti",
                            "UCSF-PDGM-0136_nifti",
                            "UCSF-PDGM-0035_nifti"]

    # getting all the filepaths from the provided data_dir
    modalities_filepaths = fop.get_patients_filepaths(data_dir,
                                                      paths_from_local_dirs,
                                                      shuffle_local_dirs=True,
                                                      n_patients=n_patients,
                                                      )

    # filtering the modalities
    logging.info(f"Filtering based on the `filtered_patients`...")

    filtered_modalities_filepaths = fop.filter_filepaths(modalities_filepaths, nonorm_test_patients)

    logging.info(f"After filtering there are {len(list(modalities_filepaths.values())[0])} filepaths remaing.")
    logging.debug(f"`modalities_filepaths`={modalities_filepaths}")

    # normalizing
    data_normalization.normalize_all_from_dir(filtered_modalities_filepaths,
                                              output_dir,
                                              normalizers,
                                              not_normalize=["mask"],
                                              divide_dataset=False
                                              )
