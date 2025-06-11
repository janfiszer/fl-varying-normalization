import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.cm as cm

from src.utils.files_operations import get_youngest_dir
from configs import config
from src.deep_learning.models import UNet
from src.deep_learning.datasets import SegmentationDataset2DSlices
from src.deep_learning.metrics import LossGeneralizedTwoClassDice, GeneralizedTwoClassDice


def load_model(unet, model_path, device):
    try:
        logging.info(f"Loading model from: {model_path}")
        unet.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        return True
    except FileNotFoundError:
        logging.error(f"You are in {os.getcwd()} and there is no given path")
        return False


def normalize_image_for_display(image):
    """Normalize image to 0-1 range for display"""
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        return (image - img_min) / (img_max - img_min)
    return image


def load_and_evaluate_model(model_path):
    pass


def evaluate_models_on_dataset(models_path, dataloader, device):
    """
    Evaluate all models on the dataset and collect results
    """
    results = {}

    # Initialize metric calculator
    dice_metric = GeneralizedTwoClassDice()

    # Evaluate each model
    for model_idx, model_path in enumerate(models_path):
        model_name = get_youngest_dir(model_path).split('-')[1]
        results[model_name] = {
            'predictions': [],
            'dice_scores': [],
            'inputs': [],
            'targets': []
        }

        # Load model
        criterion = LossGeneralizedTwoClassDice(device)
        unet = UNet(criterion)
        unet.to(device)

        if not load_model(unet, model_path, device):
            continue

        unet.eval()

        # Process each sample
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Get predictions
                outputs = unet(inputs)
                predictions = outputs
                # predictions = torch.sigmoid(outputs) > 0.5

                # Calculate dice score
                dice_score = dice_metric(predictions.float(), targets.float())

                # Store results
                results[model_name]['predictions'].append(predictions.cpu().numpy())
                results[model_name]['dice_scores'].append(dice_score.item())
                results[model_name]['inputs'].append(inputs.cpu().numpy())
                results[model_name]['targets'].append(targets.cpu().numpy())

    return results


def create_visualization(results, dataloader, binarize_pred=True, target_on_pred=False):
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
                             figsize=(3 * total_cols, 3 * total_rows))

    if total_rows == 1:
        axes = axes.reshape(1, -1)
    if total_cols == 1:
        axes = axes.reshape(-1, 1)

    cmap_input = 'gray'
    cmap_mask = 'Reds'
    if target_on_pred:
        color_list = [(1, 1, 1, 0), # fully transparent
                      (1, 0, 1, 1),  # purple
                      (1, 1, 0, 1),
                      (0, 1, 0, 1)]
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
            im = ax.imshow(img, cmap=cmap_input, vmin=vmin, vmax=vmax)

            if sample_idx == 0:
                ax.set_ylabel(f'{modality}', fontsize=10, fontweight='bold')
            axis_off_keep_ylabel(ax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            row_idx += 1

        # Target
        ax_target = axes[row_idx, sample_idx]
        target_img = target_data[0] if len(target_data.shape) == 3 else target_data
        im_target = ax_target.imshow(target_img, cmap=cmap_mask, vmin=0, vmax=1)
        if sample_idx == 0:
            ax_target.set_ylabel('Target', fontsize=10, fontweight='bold')
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

            im_pred = ax_pred.imshow(img, cmap=cmap_input, alpha=0.1)
            im_pred = ax_pred.imshow(pred_img, cmap=cmap_pred, interpolation='none')

            if sample_idx == 0:
                ax_pred.set_ylabel(model_name, fontsize=10, fontweight='bold')

            ax_pred.text(0.05, 0.95, f'Dice: {dice_score:.3f}',
                         transform=ax_pred.transAxes, fontsize=8,
                         verticalalignment='top', bbox=dict(boxstyle='round',
                                                            facecolor='white', alpha=0.8))
            axis_off_keep_ylabel(ax_pred)
            # if model_idx == num_models - 1:
            #     plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

            row_idx += 1

        # Add sample label on top
        sample_name = get_youngest_dir(dataloader.dataset.target_filepaths[sample_idx]).split('_')[0]
        axes[0, sample_idx].set_title(config.OFFICIAL_NORMALIZATION_NAMES[sample_name],
                                      fontsize=12, fontweight='bold')

    plt.tight_layout()
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
    data_dir = "C:\\Users\\JanFiszer\\data\\mri\\fl-varying-norm\\same_slice_different_norm"
    batch_size = 1

    dataset = SegmentationDataset2DSlices(data_dir, config.USED_MODALITIES, config.MASK_DIR, binarize_mask=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Loading models
    device = 'cpu'
    trained_model_dir_path = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\trained_models\\global"
    model_dirs = config.PRESENTED_MODEL_DIRs
    model_dirs.remove("model-fedbn-lr0.001-rd32-ep2-2025-05-22")
    models_path = [os.path.join(trained_model_dir_path, model_dir, "best_model.pth")
                   for model_dir in model_dirs]

    print(f"Found {len(models_path)} models to evaluate:")
    for i, path in enumerate(models_path):
        print(f"  Model {i + 1}: {path}")

    # Evaluate all models on the dataset
    print("\nEvaluating models on dataset...")
    results = evaluate_models_on_dataset(models_path, dataloader, device)

    # Print summary statistics
    print_summary_statistics(results)

    # Create visualization
    print("\nCreating visualization...")
    fig = create_visualization(results, dataloader, target_on_pred=True)

    # Save the plot
    output_path = "model_evaluation_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Show the plot
    plt.show()

    print(f"\nVisualization complete! Evaluated {len(results)} models on {len(dataloader)} samples.")