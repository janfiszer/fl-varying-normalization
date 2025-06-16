import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Patch
import matplotlib.cm as cm

from src.utils.files_operations import get_youngest_dir, sort_by_substring_order
from configs import config
from src.deep_learning.models import UNet
from src.deep_learning.datasets import SegmentationDataset2DSlices
from src.deep_learning.metrics import LossGeneralizedTwoClassDice, GeneralizedTwoClassDice


def load_model_weigths(unet, model_path, device):
    try:
        logging.info(f"Loading model from: {model_path}")
        unet.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        return True
    except FileNotFoundError:
        logging.error(f"You are in {os.getcwd()} and there is no given path")
        return False


def load_and_evaluate_model(model_dir_path, dataloader, device):
    # Initialize metric calculator
    dice_metric = GeneralizedTwoClassDice()
    criterion = LossGeneralizedTwoClassDice(device)
    fedbn_model = "fedbn" in model_dir_path
    if fedbn_model:
        client_model_paths = [os.path.join(model_dir_path, client_dir, "best_model.pth")
                              for client_dir in os.listdir(model_dir_path) if "client" in client_dir]
        personalized_unets = {}
        for client_model_path in client_model_paths:
            unet = UNet(criterion)
            unet.to(device)
            load_model_weigths(unet, client_model_path, device)
            unet.eval()
            personalized_unets[get_youngest_dir(client_model_path).split('_')[1]] = unet
    else:
        model_path = os.path.join(model_dir_path, "best_model.pth")
        # Load model
        unet = UNet(criterion)
        unet.to(device)
        unet.eval()

        if not load_model_weigths(unet, model_path, device):
            return None

    results = {
        'predictions': [],
        'dice_scores': [],
        'inputs': [],
        'targets': []
    }

    # Process each sample
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            current_slice_name = get_youngest_dir(dataloader.dataset.target_filepaths[batch_idx])
            if fedbn_model:
                unet = personalized_unets[current_slice_name.split('_')[0]]
                print(f"Model for {current_slice_name.split('_')[0]}")
            # Get predictions
            outputs = unet(inputs)
            predictions = outputs
            # predictions = torch.sigmoid(outputs) > 0.5

            # Calculate dice score
            dice_score = dice_metric(predictions.float(), targets.float())

            # Store results
            results['predictions'].append(predictions.cpu().numpy())
            results['dice_scores'].append(dice_score.item())
            results['inputs'].append(inputs.cpu().numpy())
            results['targets'].append(targets.cpu().numpy())

            print(f"For slice {current_slice_name} the GDS: {dice_score.item():.3f}")
    return results


def evaluate_models_on_dataset(model_dir_paths, dataloader, device):
    """
    Evaluate all models on the dataset and collect results
    """
    results = {}

    # Evaluate each model
    for model_idx, model_dir in enumerate(model_dir_paths):
        model_name = os.path.basename(model_dir).split('-')[1]
        models_results = load_and_evaluate_model(model_dir, dataloader, device)

        results[model_name] = models_results

    return results


def process_slice(img_np, rotate_deg=270, crop_margin=40):
    img_np = np.rot90(img_np, k=rotate_deg // 90)
    img_np = img_np[crop_margin:-crop_margin, crop_margin:-crop_margin]

    return img_np


def create_visualization(results, dataloader, binarize_pred=True, target_on_pred=False, fontsize_scaler=2):
    """
    Create comprehensive visualization with models as rows and samples as columns.
    """
    num_models = len(results)
    num_samples = len(dataloader)
    num_modalities = len(config.USED_MODALITIES)

    # Each column will now be: input modalities + target + all model predictions
    total_rows = num_modalities + 1 + num_models
    total_cols = num_samples

    fig, axes = plt.subplots(total_rows, total_cols,
                             figsize=(4 * total_cols, 3 * total_rows))

    if total_rows == 1:
        axes = axes.reshape(1, -1)
    if total_cols == 1:
        axes = axes.reshape(-1, 1)

    cmap_input = 'gray'
    cmap_mask = ListedColormap([(1, 1, 1, 0),  # fully transparent
                                (0.7, 0.5, 0.5, 1)])  # random artistic vision
    if target_on_pred:
        color_list = [(1, 1, 1, 0),  # fully transparent
                      (0.7, 0.2, 0.7, 1),  # purple
                      (0.9, 0.9, 0.1, 1),  # yellow
                      (0.2, 0.8, 0, 1)]  # green
        # reduced_color_list = [v - 0.1 for v in color_list if v == 1]
        # color_list = ['white', 'purple', 'yellow', 'green']
        cmap_pred = ListedColormap(color_list)
    else:
        cmap_pred = 'Blues'

    model_names = list(results.keys())

    for sample_idx in range(num_samples):
        first_model = model_names[0]
        input_data = results[first_model]['inputs'][sample_idx][0]
        target_data = results[first_model]['targets'][sample_idx][0]

        row_idx = 0

        # Input modalities
        for mod_idx, modality in enumerate(config.USED_MODALITIES):
            ax = axes[row_idx, sample_idx]
            img = input_data[mod_idx] if len(input_data.shape) == 3 else input_data
            vmin, vmax = img.min(), img.max()
            im = ax.imshow(process_slice(img), cmap=cmap_input, vmin=vmin, vmax=vmax)

            if sample_idx == 0:
                ax.set_ylabel(f'{config.OFFICIAL_MODALITIES_NAMES[modality]}',
                              fontsize=fontsize_scaler * 10, fontweight='bold')
            axis_off_keep_ylabel(ax)
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
            cbar.ax.tick_params(labelsize=fontsize_scaler*8)
            row_idx += 1

        # Target
        ax_target = axes[row_idx, sample_idx]
        target_img = target_data[0] if len(target_data.shape) == 3 else target_data
        ax_target.imshow(process_slice(img), cmap=cmap_input, alpha=0.1)
        im_target = ax_target.imshow(process_slice(target_img), cmap=cmap_mask, vmin=0, vmax=1)

        if sample_idx == 0:
            ax_target.set_ylabel('Target', fontsize=fontsize_scaler * 10, fontweight='bold')
            if target_on_pred:
                legend_elements = [Patch(facecolor=color, label=label) for color, label in
                                   zip(color_list[1:], ["False negative", "False positive", "True Positive"])]
                ax_target.legend(handles=legend_elements, fontsize=12*fontsize_scaler)

        axis_off_keep_ylabel(ax_target)
        row_idx += 1

        # Model predictions
        for model_idx, model_name in enumerate(model_names):
            model_results = results[model_name]
            pred_data = model_results['predictions'][sample_idx][0]
            dice_score = model_results['dice_scores'][sample_idx]
            ax_pred = axes[row_idx, sample_idx]
            pred_img = pred_data[0] if len(pred_data.shape) == 3 else pred_data
            if binarize_pred:
                pred_img = (pred_img > 0.5).astype(int)

            if target_on_pred:
                pred_img *= 2
                pred_img += target_img

            im_pred = ax_pred.imshow(process_slice(img), cmap=cmap_input, alpha=0.1)
            im_pred = ax_pred.imshow(process_slice(pred_img), cmap=cmap_pred, interpolation='none')#, vmax=3)

            if sample_idx == 0:
                ax_pred.set_ylabel(config.OFFICIAL_MODEL_NAMES[model_name], fontsize=fontsize_scaler * 10,
                                   fontweight='bold')

            ax_pred.text(0.4, 0.15, f'Dice: {dice_score:.3f}',
                         transform=ax_pred.transAxes, fontsize=fontsize_scaler * 8,
                         verticalalignment='top', bbox=dict(boxstyle='round',
                                                            facecolor='white', alpha=0.8))
            axis_off_keep_ylabel(ax_pred)

            row_idx += 1

        # Add sample label on top
        sample_name = get_youngest_dir(dataloader.dataset.target_filepaths[sample_idx]).split('_')[0]
        axes[0, sample_idx].set_title(config.OFFICIAL_NORMALIZATION_NAMES[sample_name],
                                      fontsize=fontsize_scaler * 12, fontweight='bold')

    plt.subplots_adjust(hspace=0.01, wspace=0.05)
    return fig


def axis_off_keep_ylabel(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def print_summary_statistics(results):
    """Print summary statistics for all models"""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for model_name, model_results in results.items():
        dice_scores = model_results['dice_scores']
        print(f"\n{model_name}:")
        print(f"  Mean Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
        print(f"  Min Dice Score:  {np.min(dice_scores):.4f}")
        print(f"  Max Dice Score:  {np.max(dice_scores):.4f}")
        print(f"  Median Dice Score: {np.median(dice_scores):.4f}")


if __name__ == "__main__":
    # Loading presentation dataset
    data_dir = "C:\\Users\\JanFiszer\\data\\mri\\fl-varying-norm\\same_slice_different_norm\\raw_bad"
    batch_size = 1

    dataset = SegmentationDataset2DSlices(data_dir, config.USED_MODALITIES, config.MASK_DIR, binarize_mask=True)

    # Probably something cursed to do, but I force the order of the slices for nicer visualization purpuses
    dataset.target_filepaths = sort_by_substring_order(dataset.target_filepaths, config.NORMALIZATION_ORDER)
    for modality in config.USED_MODALITIES:
        dataset.modalities_filepaths[modality] = sort_by_substring_order(dataset.modalities_filepaths[modality],
                                                                         config.NORMALIZATION_ORDER)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Loading models
    device = 'cpu'
    trained_model_dir_path = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\trained_models"
    model_dirs = config.PRESENTED_MODEL_DIRs
    model_dir_paths = [os.path.join(trained_model_dir_path, model_dir)
                       for model_dir in model_dirs]

    print(f"Found {len(model_dir_paths)} models to evaluate:")
    for i, path in enumerate(model_dir_paths):
        print(f"  Model {i + 1}: {path}")

    # Evaluate all models on the dataset
    print("\nEvaluating models on dataset...")
    results = evaluate_models_on_dataset(model_dir_paths, dataloader, device)

    # Print summary statistics
    print_summary_statistics(results)

    # Create visualization
    print("\nCreating visualization...")
    fig = create_visualization(results, dataloader, target_on_pred=True)

    # Save the plot
    output_path = "preds_visualization.svg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Show the plot
    plt.show()

    print(f"\nVisualization complete! Evaluated {len(results)} models on {len(dataloader)} samples.")
