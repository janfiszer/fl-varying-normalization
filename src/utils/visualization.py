from typing import List

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
                v_max = None
            elif col == num_modalities:
                cmap = "plasma"
                img = target
                v_max = 1
            else:
                cmap = "plasma"
                img = prediction
                v_max = 1

            if not isinstance(img, torch.Tensor):
                raise TypeError("All images, targets, and predictions must be torch.Tensor instances.")

            # Convert to NumPy, squeeze and rotate if needed
            img_np = img.numpy()

            if len(img_np.shape) > 2:
                img_np = np.squeeze(img_np)

            if rotate_deg != 0:
                img_np = np.rot90(img_np, k=rotate_deg // 90)

            ax.imshow(img_np, cmap=cmap, vmax=v_max)
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
    axs[1,0].set_ylabel('2ClassDice')
    axs[1,0].plot(epochs, network_history['two_class_generalized_dice'])
    axs[1,0].plot(epochs, network_history['val_two_class_generalized_dice'])

    axs[1,1].set_xlabel('Epochs')
    axs[1,1].set_ylabel('Torchmetrics DICE')
    axs[1,1].plot(epochs, network_history['generalized_dice_torchmetrics'])
    axs[1,1].plot(epochs, network_history['val_generalized_dice_torchmetrics'])

    axs[0,1].set_xlabel('Epochs')
    axs[0,1].set_ylabel('Function Dice')
    axs[0,1].plot(epochs, network_history['old_dice_generalized'])
    axs[0,1].plot(epochs, network_history['val_old_dice_generalized'])

    plt.legend(['Training', 'Validation'])

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()
