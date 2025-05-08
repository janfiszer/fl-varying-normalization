from typing import List
import os

from src.utils.files_operations import get_patients_filepaths
from configs import config

import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_all_modalities_and_target(
        images_list,  # List[List[torch.Tensor]]: each sample has 3 modality images (2D tensors)
        targets_list,  # List[torch.Tensor]: each is a 2D tensor (target)
        predictions_list=None,  # Optional[List[torch.Tensor]]: if provided, same length as images_list
        title=None,  # str: title in plt.title
        column_names=None,  # List[str]: optional, len = num_modalities + 1 (or +2 if predictions are included)
        row_names=None,  # List[str]: optional, len = number of samples
        rotate_deg=270,  # int or float: degrees to rotate images (applied to all)
        savepath=None
):
    num_samples = len(images_list)
    num_modalities = len(images_list[0])

    if not all(len(modalities) == num_modalities for modalities in images_list):
        raise ValueError("Each sample must have the same number of modalities.")

    if predictions_list is not None and len(predictions_list) != num_samples:
        raise ValueError("predictions_list must have the same length as images_list.")

    include_predictions = predictions_list is not None
    total_columns = num_modalities + 1 + int(include_predictions)  # modalities + target [+ prediction]

    fig, axes = plt.subplots(num_samples, total_columns, figsize=(4 * total_columns, 4 * num_samples + 2))

    # Ensure axes is always 2D
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    if total_columns == 1:
        axes = axes[:, np.newaxis]

    for row in range(num_samples):
        sample_modalities = images_list[row]
        target = targets_list[row]
        prediction = predictions_list[row] if include_predictions else None

        for col in range(total_columns):
            ax = axes[row, col]

            # Determine which image to display
            if col < num_modalities:
                img = sample_modalities[col]
                cmap = "gray"
                vmax = None
            elif col == num_modalities:
                cmap = "plasma"
                img = target
                vmax = 1
            else:
                cmap = "plasma"
                img = prediction
                vmax = 1

            if not isinstance(img, torch.Tensor):
                raise TypeError("All images, targets, and predictions must be torch.Tensor instances.")

            # Convert to NumPy, squeeze and rotate if needed
            img_np = img.numpy()

            if len(img_np.shape) > 2:
                img_np = np.squeeze(img_np)

            if rotate_deg != 0:
                img_np = np.rot90(img_np, k=rotate_deg // 90)

            ax.imshow(img_np, cmap=cmap, vmax=vmax)
            ax.axis('off')

            # Add column titles
            if row == 0 and column_names:
                if col < len(column_names):
                    ax.set_title(column_names[col], fontsize=12)

            # Add row titles
            if col == 0 and row_names:
                ax.set_ylabel(row_names[row], fontsize=12, rotation=0, labelpad=40, va='center')

    plt.tight_layout()

    if title:
        fig.suptitle(title)

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()


def plot_learning_curves(loss_histories, labels, colors=None, linetypes=None, title=None, ylabel="Loss value (MSE+DSSIM)", xlabel="Global rounds", ylim=None, figsize=None, legend=True, savepath=None, markers=None, linewidths=None, gridstyle=None, markersize=3, yscale="linear", xticks=None, yticks=None):
    if figsize:
        plt.figure(figsize=figsize)

    if ylim:
        plt.ylim(ylim)

    linestyle = '-'
    marker = None
    linewidth = 2

    for index, loss_values in enumerate(loss_histories):
        epochs = range(1, len(loss_values) + 1)

        label=labels[index]

        if linetypes:
            linestyle = linetypes[index]
        if markers:
            marker = markers[index]
        if linewidths:
            linewidth = linewidths[index]

        if colors:
            plt.plot(epochs, loss_values, label=label, linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth, color=colors[index])
        else:
            plt.plot(epochs, loss_values, label=label, linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth)

    if title is not None:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    if legend:
        plt.legend()

    if gridstyle:
        plt.grid(linestyle=gridstyle)

    if savepath:
        plt.savefig(savepath)
        plt.close()


def plot_history(network_history, savepath=None):
    epochs = range(len(network_history["loss"]))

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0,0].set_xlabel('Epochs')
    axs[0,0].set_ylabel('Loss')
    axs[0,0].plot(epochs, network_history['loss'])
    axs[0,0].plot(epochs, network_history['val_loss'])

    axs[1,0].set_xlabel('Epochs')
    axs[1,0].set_ylabel('Generalized Dice')
    axs[1,0].plot(epochs, network_history['gen_dice'])
    axs[1,0].plot(epochs, network_history['val_gen_dice'])

    axs[1,1].set_xlabel('Epochs')
    axs[1,1].set_ylabel('Dice')
    axs[1,1].plot(epochs, network_history['binarized_smoothed_dice'])
    axs[1,1].plot(epochs, network_history['val_binarized_smoothed_dice'])

    axs[0,1].set_xlabel('Epochs')
    axs[0,1].set_ylabel('Jaccard Index')
    axs[0,1].plot(epochs, network_history['binarized_jaccard_index'])
    axs[0,1].plot(epochs, network_history['val_binarized_jaccard_index'])

    plt.legend(['Training', 'Validation'])

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()


def visualize_normalization_methods(data_dir, savefig_filename=None):
    """
    Visualize histograms of MRI intensity values across different normalization methods,
    with multiple patient scans overlaid in each plot.
    """
    # Get the filepaths for each normalization method using your existing function
    # The function should return a dictionary or similar structure
    # where keys are normalization methods and values are lists of file paths
    norm_filepaths = {}
    from_local_dir_path = config.MODALITIES_AND_NPY_PATHS_FROM_LOCAL_DIR
    from_local_dir_path.pop(config.MASK_DIR)
    for normalization in os.listdir(data_dir):

        norm_filepaths[normalization] = get_patients_filepaths(os.path.join(data_dir, normalization),
                                                               config.MODALITIES_AND_NPY_PATHS_FROM_LOCAL_DIR)

    methods = list(norm_filepaths.keys())
    fig, axes = plt.subplots(len(from_local_dir_path), len(methods), figsize=(20, 16))

    x_ranges = {
        "t1":
            {
                'fcm': (0, 2),
                'minmax': (0, 1),
                'nonorm': (0, 5000),
                'nyul': (-0.5, 2),
                'whitestripe': (-0.2, 1.5),
                'zscore': (-0.5, 3)  # Adjust name and range for your 6th method
            },
        "t2":
            {
                'fcm': (0, 4),
                'minmax': (0, 1),
                'nonorm': (0, 3000),
                'nyul': (-0.5, 2),
                'whitestripe': (-0.5, 7),
                'zscore': (-0.5, 6)  # Adjust name and range for your 6th method
            },
        "flair":
            {
                'fcm': (0, 4),
                'minmax': (0, 1),
                'nonorm': (0, 3000),
                'nyul': (-0.5, 2),
                'whitestripe': (-0.5, 4),
                'zscore': (-0.5, 5)  # Adjust name and range for your 6th method
            }
    }

    # Define dataset groups if your data is organized this way (like in the reference image)
    # If not applicable, you can simplify this part
    modalities = list(norm_filepaths.values())[0].keys()

    # For demonstration - adjust this to match your actual data organization
    # This is a placeholder assuming you have different patient groups
    # If you don't have this structure, you can simplify this code
    for col, method in enumerate(methods):
        # Get file paths for this method
        filepaths = norm_filepaths[method]

        # Process each file
        for row, modality in enumerate(modalities):
            for filepath in filepaths[modality]:
                max_density = 0
                try:
                    # Load the MRI data
                    img = np.load(filepath)

                    # Extract non-zero voxels (brain tissue)
                    brain_voxels = img[img > 0].flatten()

                    # Compute histogram
                    hist, bin_edges = np.histogram(
                        brain_voxels,
                        bins=100,
                        range=x_ranges[modality][method],
                        density=True
                    )
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    color = 'blue'
                    alpha = 0.2
                    axes[row, col].plot(bin_centers, hist, color=color, alpha=alpha, linewidth=0.8)

                    current_max = np.max(hist)
                    if current_max > max_density:
                        max_density = current_max

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    continue
                
                if modality == "t1":
                    y_lim = max_density*2
                else:
                    y_lim = max_density*1.6
                axes[row, col].set_ylim(0, y_lim)

            # Set subplot title (only for first row)
            if row == 0:
                axes[row, col].set_title(config.OFFICIAL_NORMALIZATION_NAMES[method])

            # Set axes labels
            if row == len(modalities) - 1:
                axes[row, col].set_xlabel('Gray scale intensity value')

            # Set y-label only for leftmost column
            if col == 0:
                axes[row, col].set_ylabel(f'{config.OFFICIAL_MODALITIES_NAMES[modality]}\nStandardized amount of intensity')

            # Set consistent y-axis limits across all plots

    plt.tight_layout()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
