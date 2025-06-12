from typing import List
import os
from pathlib import Path
import logging
from src.utils.files_operations import get_patients_filepaths, extract_background_pixel_value
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
    from matplotlib import rcParams
    rcParams.update({'font.size': 18})
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
    fig, axes = plt.subplots(len(from_local_dir_path), len(methods), figsize=(20, 10))

    x_ranges = {
        "t1":
            {
                'fcm': (-0.5, 2),
                'minmax': (-0.2, 1),
                'nonorm': (-200, 5000),
                'nyul': (-2, 2),
                'whitestripe': (-3, 1.5),
                'zscore': (-3, 3)  # Adjust name and range for your 6th method
            },
        "t2":
            {
                'fcm': (-0.5, 4),
                'minmax': (-0.2, 1),
                'nonorm': (-200, 2000),
                'nyul': (-2, 2),
                'whitestripe': (-4, 9),
                'zscore': (-3, 6)  # Adjust name and range for your 6th method
            },
        "flair":
            {
                'fcm': (-0.5, 4),
                'minmax': (-0.2, 1),
                'nonorm': (-200, 3000),
                'nyul': (-2, 2),
                'whitestripe': (-3, 4),
                'zscore': (-3, 5)  # Adjust name and range for your 6th method
            }
    }

    # Define dataset groups if your data is organized this way (like in the reference image)
    # If not applicable, you can simplify this part
    modalities = list(norm_filepaths.values())[0].keys()

    # For demonstration - adjust this to match your actual data organization
    # This is a placeholder assuming you have different patient groups
    # If you don't have this structure, you can simplify this code
    for col, method in enumerate(config.NORMALIZATION_ORDER):
        # Get file paths for this method
        filepaths = norm_filepaths[method]

        # Process each file
        for row, modality in enumerate(modalities):
            for filepath in filepaths[modality]:
                max_density = 0
                try:
                    # Load the MRI data
                    img = np.load(filepath)

                    # Extract brain tissue
                    background_value = extract_background_pixel_value(img)
                    brain_voxels = img[img != background_value].flatten()

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
                    axes[row, col].plot(bin_centers, hist, color=color, alpha=alpha, linewidth=1)

                    current_max = np.max(hist)
                    if current_max > max_density:
                        max_density = current_max

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    continue
                
                if modality == "t1":
                    if method == "minmax":
                        y_lim = max_density * 4
                    else:
                        y_lim = max_density*2.2
                else:
                    y_lim = max_density*1.9
                axes[row, col].set_ylim(0, y_lim)

            # Set subplot title (only for first row)
            if row == 0:
                axes[row, col].set_title(config.OFFICIAL_NORMALIZATION_NAMES[method])

            # Set axes labels
            if row == len(modalities) - 1 and col == 3:
                axes[row, col].set_xlabel('Voxel intensity value')

            # Set y-label only for leftmost column
            if col == 0:
                if row == 1:
                    axes[row, col].set_ylabel(f'Standardized amount of intensity\n{config.OFFICIAL_MODALITIES_NAMES[modality]}')
                else:
                    axes[row, col].set_ylabel(f'{config.OFFICIAL_MODALITIES_NAMES[modality]}')


            # Set consistent y-axis limits across all plots

    plt.tight_layout()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_distribution(metrics_values, histograms_dir_path):
    Path(histograms_dir_path).mkdir(exist_ok=True)
    # Plot and save histograms
    for key, values in metrics_values.items():
        plt.figure()  # Create a new figure
        plt.hist(values, bins=100, color='blue', alpha=0.7)
        plt.title(f"Histogram of {key}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        # Save to file
        output_path = os.path.join(histograms_dir_path, f"{key}_histogram.png")
        plt.savefig(output_path)
        plt.close()  # Close the figure to free up memory
        logging.debug(f"\t\t\t\tSaved histogram for {key} to {output_path}")


def plot_dice_scores_combined(all_scores, title="DICE Scores Comparison", figsize=(18, 6), jitter_width=0.7, patient_plot_zone_width=50):
    """
    Create a single scatter plot with all models on the same plot.
    Patients are aligned across models (same x-position for same patient).

    Args:
        all_scores: dict {model_name: {patient_id: dice_score}}
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)

    # Get all unique patient IDs
    all_patients = set()
    for model_scores in all_scores.values():
        all_patients.update(model_scores.keys())
    all_patients = sorted(list(all_patients))

    # Create patient ID to index mapping
    patient_to_idx = {patient: i for i, patient in enumerate(all_patients)}

    # Define colors and markers for different models
    colors = ['red', 'black', 'purple', 'blue', 'green', 'orange', 'brown', 'pink', 'purple']
    markers = ['*', 'X', 'o', '^', 'X', 'X', 'X', 'X', 'X']

    model_names = list(all_scores.keys())
    num_models = len(model_names)

    # Calculate jitter offset for each model to spread them horizontally
    if num_models > 1:
        jitter_step = jitter_width / (num_models - 1)
        jitter_start = -jitter_width / 2
    else:
        jitter_step = 0
        jitter_start = 0

    for i, model_name in enumerate(model_names):
        x_positions = []
        y_scores = []

        # Calculate jitter offset for this model
        jitter_offset = jitter_start + (i * jitter_step)

        for patient_id, score in all_scores[model_name].items():
            x_positions.append(patient_to_idx[patient_id] + jitter_offset)
            y_scores.append(score)

        # Use modulo to cycle through colors and markers if there are many models
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.scatter(x_positions, y_scores,
                    c=color, marker=marker, s=60, alpha=0.7,
                    label=model_name, edgecolors='black', linewidth=0.5)

    plt.xlabel('Patient ID')
    plt.ylabel('DICE')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.grid(axis='x', alpha=0.1, linewidth=patient_plot_zone_width, color="blue")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set y-axis limits similar to the example
    plt.ylim(0.0, 1.0)

    # Set x-axis ticks to show patient names
    plt.xticks(range(len(all_patients)), all_patients, rotation=45, ha='right')

    # If there are too many patients, show only every nth patient
    if len(all_patients) > 15:
        tick_positions = list(range(0, len(all_patients), max(1, len(all_patients) // 15)))
        tick_labels = [all_patients[i][-10:-6] for i in tick_positions]
        plt.xticks(tick_positions, tick_labels, ha='right')

    plt.tight_layout()
    plt.show()
