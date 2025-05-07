from src.utils.visualization import visualize_normalization_methods
import configs
import os

if __name__ == '__main__':
    if configs.config.LOCAL:
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\from-nonorm-normalized-UCSF-PDGM"
    else:
        data_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-PDGM-v3"

    visualize_normalization_methods(data_dir, os.path.join(data_dir, "normalization_all_histograms.png"))
