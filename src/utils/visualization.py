from typing import List

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import Tensor

import math

def plot_all_modalities_and_target(
    images_list,          # List[List[torch.Tensor]]: each sample has 3 modality images (2D tensors)
    targets_list,         # List[torch.Tensor]: each is a 2D tensor (target)
    column_names=None,    # List[str]: optional, len = num_modalities + 1 (or +2 if predictions are included)
    row_names=None,       # List[str]: optional, len = number of samples
    rotate_deg=0,         # int or float: degrees to rotate images (applied to all)
    predictions_list=None, # Optional[List[torch.Tensor]]: if provided, same length as images_list
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

    fig, axes = plt.subplots(num_samples, total_columns, figsize=(4 * total_columns, 4 * num_samples))

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
            elif col == num_modalities:
                cmap = "plasma"
                img = target
            else:
                cmap = "plasma"
                img = prediction

            if not isinstance(img, torch.Tensor):
                raise TypeError("All images, targets, and predictions must be torch.Tensor instances.")

            # Convert to NumPy and rotate if needed
            img_np = img.numpy()
            if rotate_deg != 0:
                img_np = np.rot90(img_np, k=rotate_deg // 90)

            ax.imshow(img_np, cmap=cmap)
            ax.axis('off')

            # Add column titles
            if row == 0 and column_names:
                if col < len(column_names):
                    ax.set_title(column_names[col], fontsize=12)

            # Add row titles
            if col == 0 and row_names:
                ax.set_ylabel(row_names[row], fontsize=12, rotation=0, labelpad=40, va='center')

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()
    