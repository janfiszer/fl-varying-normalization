import os
import matplotlib.pyplot as plt
import numpy as np
from src.utils import files_operations as fop
from typing import Dict
import nibabel as nib
from configs import config


def plot_and_save(data_dict, filepath):
    """
    Plots multiple numpy array images from a dictionary on a subplot grid and saves the figure.

    Parameters:
    data_dict (dict): A dictionary where keys are titles (labels) and values are numpy arrays (images).
    filepath (str): The path where the plot should be saved.
    """
    num_images = len(data_dict)
    cols = min(3, num_images)  # Maximum 3 columns
    rows = (num_images + cols - 1) // cols  # Calculate necessary rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)  # Flatten axes for easy iteration

    for ax, (title, image) in zip(axes, data_dict.items()):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    for ax in axes[num_images:]:  # Hide any unused subplots
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_slice_for_patient(data_dir, output_dir, path_from_local_dir: Dict, slice_index=110):
    all_filepaths = fop.get_nii_filepaths(data_dir, path_from_local_dir, shuffle_local_dirs=False)

    # for each modality:
    # load the data and normalize the data with the given normalizer
    # store it in a new directory
    fop.try_create_dir(output_dir)
    print("The sample slices will be save in: ",  output_dir)

    first_modality = list(all_filepaths.keys())[0]

    # for each patient: visualize each modality slice from every volume
    for path_index in range(len(all_filepaths[first_modality])):
        slices_to_visualize = {}
        full_output_path = None
        for modality in all_filepaths.keys():
            # visualizing a slice
            volume_path = all_filepaths[modality][path_index]
            patient_name = volume_path.split(os.path.sep)[-2]
            full_output_path = os.path.join(output_dir, f"{patient_name}_sample_slices{slice_index}.png")
            volume = nib.load(volume_path).get_fdata()

            slices_to_visualize[modality] = volume[:, :, slice_index]

        plot_and_save(slices_to_visualize, full_output_path)

    print("\nVisualizing ends.")


if __name__ == '__main__':
    if config.LOCAL:
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\flair_volumes"
        output_dir = "C:\\Users\\JanFiszer\\data\\mri\\flair_volumes\\single_slice_vis"
        paths_from_local_dirs = {"t1": "*t1.nii.gz", "t2": "*t2.nii.gz", "flair": "*flair.nii.gz"}
    else:
        data_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-PDGM-v3"
        output_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/UCSF-PDGM-v3/single_slice_vis"
        paths_from_local_dirs = {"t1": "*T1.nii.gz", "t2": "*T2.nii.gz", "flair": "*FLAIR.nii.gz"}
    # path_from_local_dir = {}
    plot_slice_for_patient(data_dir, output_dir, paths_from_local_dirs)
