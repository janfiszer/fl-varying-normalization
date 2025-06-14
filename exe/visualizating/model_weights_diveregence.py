import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import math
from src.deep_learning import models

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import math
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from configs import config

def load_model_and_plot_weight_histograms(model_path, model_class, model_kwargs=None,
                                          figsize_per_plot=(6, 4), cols=3,
                                          bins=50, save_path=None):
    """
    Load a PyTorch model and plot histograms for each layer's weights.

    Args:
        model_path (str): Path to the saved model (.pth or .pt file)
        model_class: The model class (e.g., UNet)
        model_kwargs (dict): Keyword arguments to initialize the model
        figsize_per_plot (tuple): Size of each individual histogram plot
        cols (int): Number of columns in the subplot grid
        bins (int): Number of bins for histograms
        save_path (str): Optional path to save the figure
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Initialize the model
    model = model_class(**model_kwargs)

    # Load the state dict
    try:
        state_dict = torch.load(model_path, map_location='cpu')

        # Handle cases where state_dict might be wrapped in a dictionary
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get all parameters with weights (excluding biases and normalization parameters)
    weight_layers = OrderedDict()

    for name, param in model.named_parameters():
        # Focus on convolutional and linear layer weights
        if 'norm' in name and 'weight' in name:
            # if 'norm' in name and param.dim() >= 2:
            # Skip normalization layer weights (they're typically small and not as interesting)
            # if not any(norm_type in name for norm_type in ['norm', 'bn', 'batch_norm', 'group_norm']):
            weight_layers[name] = param.detach().cpu().numpy().flatten()

    if not weight_layers:
        print("No weight parameters found in the model!")
        return

    # Calculate subplot grid dimensions
    n_layers = len(weight_layers)
    rows = math.ceil(n_layers / cols)

    # Create the figure
    fig_width = cols * figsize_per_plot[0]
    fig_height = rows * figsize_per_plot[1]
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Ensure axes is always 2D for consistent indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot histograms
    for idx, (layer_name, weights) in enumerate(weight_layers.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Plot histogram
        ax.hist(weights, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Customize the plot
        ax.set_title(f'{layer_name}\n({len(weights):,} params)', fontsize=10, pad=10)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(weights)
        std_val = np.std(weights)
        ax.text(0.02, 0.98, f'μ={mean_val:.4f}\nσ={std_val:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)

    # Hide empty subplots
    for idx in range(n_layers, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print(f"\nModel Weight Statistics Summary:")
    print(f"{'Layer Name':<30} {'Params':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)

    for layer_name, weights in weight_layers.items():
        print(f"{layer_name:<30} {len(weights):<10,} {np.mean(weights):<10.4f} "
              f"{np.std(weights):<10.4f} {np.min(weights):<10.4f} {np.max(weights):<10.4f}")


def quick_histogram_analysis(model_path, model_class, model_kwargs=None):
    """
    Quick analysis function that provides a summary without plotting.
    Useful for getting a quick overview of weight distributions.
    """
    if model_kwargs is None:
        model_kwargs = {}

    model = model_class(**model_kwargs)

    try:
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Weight Layer Analysis:")
    print("=" * 60)

    total_params = 0
    for name, param in model.named_parameters():
        if 'norm' in name and 'weight' in name:
            # if 'weight' in name and param.dim() >= 2:
            # if not any(norm_type in name for norm_type in ['norm', 'bn', 'batch_norm', 'group_norm']):
            weights = param.detach().cpu().numpy().flatten()
            total_params += len(weights)

            print(f"\nLayer: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Parameters: {len(weights):,}")
            print(f"  Mean: {np.mean(weights):.6f}")
            print(f"  Std: {np.std(weights):.6f}")
            print(f"  Range: [{np.min(weights):.6f}, {np.max(weights):.6f}]")

    print(f"\nTotal weight parameters analyzed: {total_params:,}")


def compute_weight_histograms(model_path, model_class, model_kwargs=None, bins=50):
    """
    Load a model and compute histograms for each weight layer.
    Returns a dictionary with layer names as keys and (bin_edges, hist_values) as values.
    """
    if model_kwargs is None:
        model_kwargs = {}

    model = model_class(**model_kwargs)

    try:
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

    layer_histograms = OrderedDict()

    for name, param in model.named_parameters():
        # if 'weight' in name and param.dim() >= 2:
        if 'norm' in name and 'weight' in name:
            # if not any(norm_type in name for norm_type in ['norm', 'bn', 'batch_norm', 'group_norm']):
            weights = param.detach().cpu().numpy().flatten()

            # Compute histogram with fixed range for consistency across models
            weights_min, weights_max = np.min(weights), np.max(weights)
            # Use a slightly expanded range to ensure all values are captured
            range_min = weights_min - 0.1 * abs(weights_min)
            range_max = weights_max + 0.1 * abs(weights_max)

            hist_values, bin_edges = np.histogram(weights, bins=bins,
                                                  range=(range_min, range_max),
                                                  density=True)
            layer_histograms[name] = (bin_edges, hist_values)

    return layer_histograms


def compute_kl_divergence_safe(p, q, epsilon=1e-10):
    """
    Compute KL divergence between two probability distributions with numerical stability.
    Uses Jensen-Shannon divergence as fallback for better stability.
    """
    # Add small epsilon to avoid log(0)
    p = p + epsilon
    q = q + epsilon

    # Normalize to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
    # Use scipy's entropy function: KL(p, q) = entropy(p, q)
    try:
        kl_div = entropy(p, q)
        # If KL divergence is infinite or nan, use Jensen-Shannon divergence
        if np.isinf(kl_div) or np.isnan(kl_div):
            # Jensen-Shannon divergence is symmetric and bounded
            js_div = jensenshannon(p, q) ** 2  # Square to get JS divergence from JS distance
            return js_div
        return kl_div
    except:
        # Fallback to Jensen-Shannon divergence
        js_div = jensenshannon(p, q) ** 2
        return js_div


def plot_kl_divergences_bar_plot(reference_model_path, eval_model_paths, model_class,
                        model_kwargs=None, model_names=None, bins=50,
                        figsize=(15, 8), save_path=None):
    """
    Compare multiple models against a reference model by computing KL divergences
    of weight distributions for each layer and plotting them in a grouped bar chart.

    Args:
        reference_model_path (str): Path to the reference model
        eval_model_paths (list): List of paths to evaluation models
        model_class: The model class (e.g., UNet)
        model_kwargs (dict): Keyword arguments to initialize the model
        model_names (list): Names for the evaluation models (for legend)
        bins (int): Number of bins for histograms
        figsize (tuple): Figure size
        save_path (str): Optional path to save the figure
    """
    if model_kwargs is None:
        model_kwargs = {}

    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(eval_model_paths))]

    # Load reference model histograms
    print("Loading reference model...")
    ref_histograms = compute_weight_histograms(reference_model_path, model_class,
                                               model_kwargs, bins)
    if ref_histograms is None:
        return

    # Load evaluation model histograms
    eval_histograms = []
    for i, model_path in enumerate(eval_model_paths):
        print(f"Loading evaluation model {i + 1}/{len(eval_model_paths)}...")
        hist = compute_weight_histograms(model_path, model_class, model_kwargs, bins)
        if hist is not None:
            eval_histograms.append(hist)
        else:
            print(f"Skipping model {i + 1} due to loading error")

    if not eval_histograms:
        print("No evaluation models loaded successfully!")
        return

    # Compute KL divergences
    layer_names = list(ref_histograms.keys())
    kl_divergences = {name: [] for name in model_names[:len(eval_histograms)]}

    print("Computing KL divergences...")
    for layer_name in layer_names:
        ref_bin_edges, ref_hist = ref_histograms[layer_name]

        for i, eval_hist_dict in enumerate(eval_histograms):
            if layer_name in eval_hist_dict:
                eval_bin_edges, eval_hist = eval_hist_dict[layer_name]

                # Ensure histograms are computed over the same range
                # Recompute histograms with unified range
                all_edges = np.concatenate([ref_bin_edges, eval_bin_edges])
                unified_range = (np.min(all_edges), np.max(all_edges))

                # We need the original weights to recompute with unified range
                # For now, use the existing histograms but normalize them properly
                kl_div = compute_kl_divergence_safe(ref_hist, eval_hist)
                kl_divergences[model_names[i]].append(kl_div)
            else:
                print(f"Warning: Layer {layer_name} not found in model {i + 1}")
                kl_divergences[model_names[i]].append(np.nan)

    # Create the bar plot
    fig, ax = plt.subplots(figsize=figsize)

    n_layers = len(layer_names)
    n_models = len(eval_histograms)

    # Set up bar positions
    bar_width = 0.8 / n_models
    x_positions = np.arange(n_layers)

    # Colors for different models
    colors = plt.cm.Set1(np.linspace(0, 1, n_models))

    # Plot bars for each model
    for i, (model_name, kl_values) in enumerate(kl_divergences.items()):
        x_pos = x_positions + (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x_pos, kl_values, bar_width, label=model_name,
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, kl_val in zip(bars, kl_values):
            if not np.isnan(kl_val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{kl_val:.1f}', ha='center', va='bottom', fontsize=8)

    # Customize the plot
    ax.set_xlabel('Layer Name', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('KL Divergences of Weight Distributions\n(Reference vs Evaluation Models)',
                 fontsize=14, pad=20)

    # Set x-axis labels
    ax.set_xticks(x_positions)
    # Shorten layer names for better readability
    shortened_names = [name.replace('conv', 'c').replace('weight', 'w') for name in layer_names]
    ax.set_xticklabels(shortened_names, rotation=45, ha='right')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print(f"\nKL Divergence Summary:")
    print("=" * 60)

    for model_name, kl_values in kl_divergences.items():
        valid_kl = [kl for kl in kl_values if not np.isnan(kl)]
        if valid_kl:
            print(f"\n{model_name}:")
            print(f"  Mean KL Divergence: {np.mean(valid_kl):.4f}")
            print(f"  Std KL Divergence: {np.std(valid_kl):.4f}")
            print(f"  Max KL Divergence: {np.max(valid_kl):.4f}")
            print(f"  Min KL Divergence: {np.min(valid_kl):.4f}")

    return kl_divergences


def plot_kl_divergences(reference_model_path, eval_model_paths, model_class,
                        model_kwargs=None, model_names=None, bins=50,
                        figsize=(12, 8), save_path=None):
    """
    Compare multiple models against a reference model by computing KL divergences
    of weight distributions for each layer and plotting them in a STACKED bar chart.
    Each model gets a stacked bar where colors represent different weight layers.

    Args:
        reference_model_path (str): Path to the reference model
        eval_model_paths (list): List of paths to evaluation models
        model_class: The model class (e.g., UNet)
        model_kwargs (dict): Keyword arguments to initialize the model
        model_names (list): Names for the evaluation models (for legend)
        bins (int): Number of bins for histograms
        figsize (tuple): Figure size
        save_path (str): Optional path to save the figure
    """
    if model_kwargs is None:
        model_kwargs = {}

    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(eval_model_paths))]

    # Load reference model histograms
    print("Loading reference model...")
    ref_histograms = compute_weight_histograms(reference_model_path, model_class,
                                               model_kwargs, bins)
    if ref_histograms is None:
        return

    # Load evaluation model histograms
    eval_histograms = []
    for i, model_path in enumerate(eval_model_paths):
        print(f"Loading evaluation model {i + 1}/{len(eval_model_paths)}...")
        hist = compute_weight_histograms(model_path, model_class, model_kwargs, bins)
        if hist is not None:
            eval_histograms.append(hist)
        else:
            print(f"Skipping model {i + 1} due to loading error")

    if not eval_histograms:
        print("No evaluation models loaded successfully!")
        return

    # Compute KL divergences
    layer_names = list(ref_histograms.keys())

    # Create matrix: models x layers
    kl_matrix = np.zeros((len(eval_histograms), len(layer_names)))

    print("Computing KL divergences...")
    for j, layer_name in enumerate(layer_names):
        ref_bin_edges, ref_hist = ref_histograms[layer_name]

        for i, eval_hist_dict in enumerate(eval_histograms):
            if layer_name in eval_hist_dict:
                eval_bin_edges, eval_hist = eval_hist_dict[layer_name]
                kl_div = compute_kl_divergence_safe(ref_hist, eval_hist)
                kl_matrix[i, j] = kl_div
            else:
                print(f"Warning: Layer {layer_name} not found in model {i + 1}")
                kl_matrix[i, j] = 0.0

    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=figsize)

    # Generate colors for each layer (similar to your reference image)
    colors = plt.cm.Set3(np.linspace(0, 1, len(layer_names)))

    # Alternative color schemes you can try:
    # colors = plt.cm.tab20(np.linspace(0, 1, len(layer_names)))
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    x_positions = np.arange(len(model_names))
    bar_width = 0.6

    # Create stacked bars
    bottom = np.zeros(len(model_names))

    # Shortened layer names for legend
    shortened_layer_names = []
    for name in layer_names:
        # Extract meaningful parts of layer names
        parts = name.split('.')
        if len(parts) >= 2:
            # Take last 2 parts and abbreviate
            short_name = '.'.join(parts[-2:])
            short_name = short_name.replace('norm', 'N').replace('weight', 'W')
            short_name = short_name.replace('group_norm', 'GN').replace('batch_norm', 'BN')
        else:
            short_name = name.replace('norm', 'N').replace('weight', 'W')
        shortened_layer_names.append(short_name)

    # Plot each layer as a segment in the stacked bar
    for j, (layer_name, short_name) in enumerate(zip(layer_names, shortened_layer_names)):
        layer_values = kl_matrix[:, j]
        bars = ax.bar(x_positions, layer_values, bar_width,
                      bottom=bottom, label=short_name,
                      color=colors[j], alpha=0.8,
                      edgecolor='white', linewidth=0.5)
        bottom += layer_values

    # Customize the plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('KL Divergences of Weight Distributions\n(Stacked by Layer)',
                 fontsize=14, fontweight='bold', pad=20)

    # Set x-axis labels
    ax.set_xticks(x_positions)
    # Shorten model names if they're too long
    shortened_model_names = []
    for name in model_names:
        if len(name) > 15:
            # Take first few and last few characters
            shortened = name[:8] + '...' + name[-4:]
        else:
            shortened = name
        shortened_model_names.append(shortened)

    ax.set_xticklabels(shortened_model_names, rotation=45, ha='right')

    # Create legend outside the plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              title='Layers', title_fontsize=10, fontsize=9)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Add total values on top of each bar
    for i, (x_pos, total_kl) in enumerate(zip(x_positions, np.sum(kl_matrix, axis=1))):
        ax.text(x_pos, total_kl + 0.02, f'{total_kl:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print(f"\nKL Divergence Summary:")
    print("=" * 60)

    for i, model_name in enumerate(model_names):
        total_kl = np.sum(kl_matrix[i, :])
        print(f"\n{model_name}:")
        print(f"  Total KL Divergence: {total_kl:.4f}")
        print(f"  Average per layer: {np.mean(kl_matrix[i, :]):.4f}")
        print(f"  Layer contributions:")
        for j, (layer_name, kl_val) in enumerate(zip(shortened_layer_names, kl_matrix[i, :])):
            percentage = (kl_val / total_kl * 100) if total_kl > 0 else 0
            print(f"    {layer_name}: {kl_val:.4f} ({percentage:.1f}%)")

    return kl_matrix, layer_names, model_names


def plot_cos_similarities(reference_model_path, eval_model_paths, model_class, model_names=None, bins=50,
                        figsize=(15, 8), save_path=None):
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(eval_model_paths))]
    layer_names = None

    cos_similarities_values = {}

    print("Compute cosine similarity:")
    for i, model_path in enumerate(eval_model_paths):
        print(f"Loading evaluation model {i + 1}/{len(eval_model_paths)}...")
        cos_sim = cosine_similarity_all_layers(model_path, reference_model_path, model_class)
        cos_similarities_values[model_names[i]] = cos_sim
        layer_names = cos_sim.keys()

    cos_similarities_lists = {name: [] for name in model_names[:len(cos_similarities_values)]}

    for model_name, cos_sim_layers_dict in cos_similarities_values.items():
        for layer_name in layer_names:
            cos_similarities_lists[model_name].append(cos_sim_layers_dict[layer_name])

    # Create the bar plot
    fig, ax = plt.subplots(figsize=figsize)

    n_layers = len(layer_names)
    n_models = len(eval_model_paths)

    # Set up bar positions
    bar_width = 0.8 / n_models
    x_positions = np.arange(n_layers)

    # Colors for different models
    colors = plt.cm.Set1(np.linspace(0, 1, n_models))

    # Plot bars for each model
    for i, (model_name, kl_values) in enumerate(cos_similarities_lists.items()):
        x_pos = x_positions + (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x_pos, kl_values, bar_width, label=model_name,
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, kl_val in zip(bars, kl_values):
            if not np.isnan(kl_val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{kl_val:.1f}', ha='center', va='bottom', fontsize=8)

    # Customize the plot
    ax.set_xlabel('Layer Name', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('KL Divergences of Weight Distributions\n(Reference vs Evaluation Models)',
                 fontsize=14, pad=20)

    # Set x-axis labels
    ax.set_xticks(x_positions)
    # Shorten layer names for better readability
    shortened_names = [name.replace('conv', 'c').replace('weight', 'w') for name in layer_names]
    ax.set_xticklabels(shortened_names, rotation=45, ha='right')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print(f"\nKL Divergence Summary:")
    print("=" * 60)

    for model_name, kl_values in cos_similarities_lists.items():
        valid_kl = [kl for kl in kl_values if not np.isnan(kl)]
        if valid_kl:
            print(f"\n{model_name}:")
            print(f"  Mean KL Divergence: {np.mean(valid_kl):.4f}")
            print(f"  Std KL Divergence: {np.std(valid_kl):.4f}")
            print(f"  Max KL Divergence: {np.max(valid_kl):.4f}")
            print(f"  Min KL Divergence: {np.min(valid_kl):.4f}")

    return cos_similarities_lists


def compute_cosine_similarity(layer_weights_1, layer_weights_2):
    return np.dot(layer_weights_1, layer_weights_2) / (np.linalg.norm(layer_weights_1) * np.linalg.norm(layer_weights_2))


def cosine_similarity_all_layers(model_path, reference_model_path, model_class):
    ref_model = model_class()
    model = model_class()

    try:
        # loading model
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)

        # loading reference model
        state_dict = torch.load(reference_model_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        ref_model.load_state_dict(state_dict)

    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

    ref_model_named_params = ref_model.named_parameters()

    layer_histograms = OrderedDict()

    for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), ref_model_named_params):
        assert name == ref_name
        with torch.no_grad():
            # if 'weight' in name and param.dim() >= 2:
            if 'norm' in name and 'weight' in name:
                layer_histograms[name] = compute_cosine_similarity(param.numpy(), ref_param.numpy())

    return layer_histograms


# Example usage:
if __name__ == "__main__":
    # Example of how to use the functions

    # For your UNet model, you would call it like this:
    # load_model_and_plot_weight_histograms(
    #     model_path="path/to/your/model.pth",
    #     model_class=UNet,  # Your UNet class
    #     model_kwargs={'bilinear': False},  # Any arguments needed for UNet initialization
    #     figsize_per_plot=(5, 4),
    #     cols=4,
    #     bins=50,
    #     save_path="weight_histograms.png"
    # )

    # Or for a quick analysis without plots:
    # quick_histogram_analysis(
    #     model_path="path/to/your/model.pth",
    #     model_class=UNet,
    #     model_kwargs={'bilinear': False}
    # )
    presented_model_dirs = [model_dir for model_dir in config.PRESENTED_MODEL_DIRs if "fedbn" not in model_dir and "all" not in model_dir]

    trained_model_dir_path = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\trained_models\\global"
    models_path = [os.path.join(trained_model_dir_path, model_dir, "best_model.pth")
                   for model_dir in presented_model_dirs]

    # For KL divergence comparison:
    plot_kl_divergences(
        reference_model_path=os.path.join(trained_model_dir_path, "model-all-ep16-lr0.001-GN-2025-04-30-13h", "best_model.pth"),
        eval_model_paths=models_path,
        model_class=models.UNet,
        model_kwargs={'bilinear': False},
        model_names=presented_model_dirs,
        save_path="vis_results/kl_divergences_only_normlayers.png",
        figsize=(40, 10)
    )

    plot_cos_similarities(
        reference_model_path=os.path.join(trained_model_dir_path, "model-all-ep16-lr0.001-GN-2025-04-30-13h", "best_model.pth"),
        eval_model_paths=models_path,
        model_class=models.UNet,
        model_names=presented_model_dirs,
        save_path="vis_results/cos_similarities_only_normlayers.png",
        figsize=(40, 10)
    )



