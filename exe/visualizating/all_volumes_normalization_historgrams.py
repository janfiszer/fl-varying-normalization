from src.utils.visualization import visualize_normalization_methods
import configs
import os
import logging

if __name__ == '__main__':
    if configs.config.LOCAL:
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM"
    else:
        data_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-1ds-normalized-test"

    save_path = os.path.join(data_dir, "normalization_all_histograms.png")
    visualize_normalization_methods(data_dir, save_path)
    logging.info(f"Finished and histogram save to: {save_path}")

