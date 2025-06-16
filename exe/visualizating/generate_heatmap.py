import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import os
from PIL import Image
from scipy.ndimage import zoom
from pathlib import Path
from configs import config


def generate_table_with_std(metric_filename, std_filename, official_names_map, cmap="Spectral"):
    official_names = list(official_names_map.values())

    vmin = 0.5
    vmax = 0.83
    round_digits = 2
    col_names = official_names

    figsize = (len(col_names) * 4, 10)
    # Loading the tables
    dirpath = "tables"
    metric_filepath = os.path.join(dirpath, metric_filename)
    std_filepath = os.path.join(dirpath, std_filename)

    metric_table = pd.read_csv(metric_filepath, index_col='Unnamed: 0')
    std_table = pd.read_csv(std_filepath, index_col='Unnamed: 0')

    if "Average" not in metric_table.columns:
        mean_column = metric_table.mean(axis=1)
        metric_table["Average"] = mean_column

    # Changing the column names for the official ones
    metric_table.columns = col_names + ["Average"]
    std_table.columns = col_names

    # Combine metric values with std for annotations
    combined_table = metric_table.iloc[:, :-1].round(round_digits).astype(str) + " ± " + std_table.round(
        round_digits).astype(str)
    combined_table["Average"] = metric_table["Average"].round(round_digits).astype(str)

    plt.figure(figsize=figsize)
    heatmap = sn.heatmap(metric_table.astype(float),
                         annot=combined_table.values, fmt='s',
                         vmin=vmin, vmax=vmax,
                         annot_kws={"size": 20},
                         cmap=cmap)

    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=20, rotation=0)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=20, rotation=0)
    heatmap.xaxis.tick_top()

    # Customize colorbar font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(os.path.join(dirpath, f"{metric_filename.split('.')[0]}_with_std.svg"))


def create_heatmap_dfs(metric_filepath, std_filepath, visualizing_methods=None):
    dirpath = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\results_analysis\\both\\tables\\new_raw_model"
    metric_filepath = os.path.join(dirpath, metric_filepath)
    std_filepath = os.path.join(dirpath, std_filepath)
    metric_table = pd.read_csv(metric_filepath, index_col='Unnamed: 0')
    std_table = pd.read_csv(std_filepath, index_col='Unnamed: 0')

    average_row_name = "Dataset average"
    average_column_name = "Average"
    # adding the average dataset-wise
    transposed_metric_table = metric_table.T
    transposed_metric_table[average_row_name] = transposed_metric_table[
        config.OFFICIAL_NORMALIZATION_NAMES.values()].mean(axis=1)

    # reordering, the average column right after the dataset and filtering
    new_filtered_column_order = ["Centralized*"] + list(
        transposed_metric_table.columns[:len(config.NORMALIZATION_ORDER)]) + [average_row_name] + visualizing_methods
    filtered_transposed_metric_table = transposed_metric_table[new_filtered_column_order]

    metric_table = filtered_transposed_metric_table.T  # reverting the transpose

    if average_column_name not in metric_table.columns:
        mean_column = metric_table.mean(axis=1)
        metric_table[average_column_name] = mean_column

    filtered_std_tables = std_table.loc[
        ["Centralized*"] + list(transposed_metric_table.columns[:len(config.NORMALIZATION_ORDER)]) + visualizing_methods]

    # removing average of averages, not desired
    metric_table[average_column_name][average_row_name] = None

    # Combine metric values with std for annotations
    combined_table = metric_table.iloc[:, :-1].map(lambda x: f"{x:.2f}") + " ± " + filtered_std_tables.map(
        lambda x: f"{x:.2f}")
    combined_table = combined_table.loc[metric_table.index]
    combined_table[average_column_name] = metric_table[average_column_name].map(lambda x: f"{x:.3f}")
    combined_table.loc[average_row_name] = metric_table.loc[average_row_name].map(lambda x: f"{x:.3f}")

    return combined_table, metric_table


def insert_gap_row(annotation_df, metric_df, gap_index, gap_row_name=""):
    """
    Inserts an empty row (NaN for metrics, empty strings for annotations) into the given DataFrames at a specific index.

    Parameters:
        metric_df (pd.DataFrame): DataFrame containing metric values (numeric).
        annotation_df (pd.DataFrame): DataFrame containing annotations (strings).
        gap_index (int): The row index after which the gap should be inserted.

    Returns:
        tuple: (metric_df_with_gap, annotation_df_with_gap) with the gap inserted.
    """
    # Create the gap rows
    gap_row_metric = pd.DataFrame([[np.nan] * metric_df.shape[1]], columns=metric_df.columns, index=[gap_index])
    gap_row_annotation = pd.DataFrame([[""] * annotation_df.shape[1]], columns=annotation_df.columns, index=[gap_index])

    # Insert the gap row into the DataFrames at the specified position
    metric_df_with_gap = pd.concat(
        [metric_df.iloc[:gap_index], gap_row_metric, metric_df.iloc[gap_index:]]
    )
    annotation_df_with_gap = pd.concat(
        [annotation_df.iloc[:gap_index], gap_row_annotation, annotation_df.iloc[gap_index:]]
    )

    # Preserve the original indices
    metric_df_with_gap.index = pd.Index(
        list(metric_df.index[:gap_index]) + [gap_row_name] + list(metric_df.index[gap_index:]))
    annotation_df_with_gap.index = pd.Index(
        list(annotation_df.index[:gap_index]) + [gap_row_name] + list(annotation_df.index[gap_index:]))

    return annotation_df_with_gap, metric_df_with_gap


def single_heatmap(metric_file_name,
                   title,
                   official_names_map,
                   font_size=23,
                   cmap="gist_rainbow",
                   methods_to_visualize=None):
    metric_filename_top = f"metrics_{metric_file_name}.csv"
    std_filename_top = f"std_{metric_file_name}.csv"
    official_names = list(official_names_map.values())

    if metric_file_name == "val_gen_dice":
        vmin = 0.2
        vmax = 0.95
    else:
        vmin = 0.2
        vmax = 0.95

    col_names = official_names

    figsize = (len(col_names) * 4, 9)

    # create the dfs needed for heatmaps
    combined_table_top, metric_table_top = create_heatmap_dfs(metric_filename_top, std_filename_top,
                                                              visualizing_methods=methods_to_visualize)
    # Create gap rows (filled with NaN for metrics and empty strings for annotations)
    combined_table_top, metric_table_top = insert_gap_row(combined_table_top, metric_table_top, gap_index=8,
                                                          gap_row_name="Federated learning:")
    combined_table_top, metric_table_top = insert_gap_row(combined_table_top, metric_table_top, gap_index=1,
                                                          gap_row_name="Single-dataset trained:")
    combined_table_top, metric_table_top = insert_gap_row(combined_table_top, metric_table_top, gap_index=12,
                                                          gap_row_name="Tested on:")

    # Create a figure and gridspec for fine-tuned layout
    fig = plt.figure(figsize=figsize)
    heatmap = sn.heatmap(metric_table_top.astype(float),
                         annot=combined_table_top.values, fmt='s',
                         vmin=vmin, vmax=vmax,
                         annot_kws={"size": font_size},
                         cmap=cmap)

    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=font_size, rotation=0)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=font_size, rotation=0)
    heatmap.set_title("Tested on dataset:", fontdict={'fontsize': font_size + 10}, pad=15)

    # plt.hlines(1, 0, 8.0, color="white", lw=4)
    plt.hlines(8, 0, 8.0, color="white", lw=4)
    plt.vlines(6, 0, 14.0, color="white", lw=4)

    heatmap.xaxis.tick_top()
    plt.title(title, fontdict={'fontsize': font_size + 10}, pad=15)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)

    dirpath = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\results_analysis\\both\\heatmaps"

    # Save the created plot
    Path(dirpath).mkdir(exist_ok=True)
    output_filepath = os.path.join(dirpath, f"{metric_file_name}.svg")

    plt.savefig(os.path.join(dirpath, output_filepath))
    print("Saved to: ", output_filepath)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


if __name__ == "__main__":
    methods_to_present = ["FedAvg", "FedBN"]

    single_heatmap("val_gen_dice",
                   "Tested on:   Trained on:  Models:  Testsets:  Single-dataset trained:  Federated learning:",
                   official_names_map={"nonorm": "Raw", "minmax": "MinMax", "zscore": "Z-Score", "nyul": "Nyul",
                                       "fcm": "Fuzzy C-Mean", "whitestripe": "WhiteStripe"},
                   cmap="Spectral",
                   methods_to_visualize=methods_to_present)
