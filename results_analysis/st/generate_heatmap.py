import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import os
from PIL import Image
from scipy.ndimage import zoom
from pathlib import Path


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


def create_heatmap_dfs(metric_filepath, std_filepath, round_digits):
    dirpath = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\results_analysis\\st\\tables"
    metric_filepath = os.path.join(dirpath, metric_filepath)
    std_filepath = os.path.join(dirpath, std_filepath)
    metric_table = pd.read_csv(metric_filepath, index_col='Unnamed: 0')
    std_table = pd.read_csv(std_filepath, index_col='Unnamed: 0')

    if "Average" not in metric_table.columns:
        mean_column = metric_table.mean(axis=1)
        metric_table["Average"] = mean_column

    # Combine metric values with std for annotations
    combined_table = metric_table.iloc[:, :-1].map(lambda x: f"{x:.2f}") + " ± " + std_table.map(lambda x: f"{x:.2f}")
    combined_table["Average"] = metric_table["Average"].map(lambda x: f"{x:.2f}")

    return combined_table, metric_table


def insert_gap_row(annotation_df, metric_df, gap_index):
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
    metric_df_with_gap.index = pd.Index(list(metric_df.index[:gap_index]) + [""] + list(metric_df.index[gap_index:]))
    annotation_df_with_gap.index = pd.Index(
        list(annotation_df.index[:gap_index]) + [""] + list(annotation_df.index[gap_index:]))

    return annotation_df_with_gap, metric_df_with_gap


def single_heatmap(metric_file_name,
                   title,
                   official_names_map,
                   image_path=None,
                   font_size=20,
                   cmap="gist_rainbow"):

    metric_filename_top = f"metrics_{metric_file_name}.csv"
    std_filename_top = f"std_{metric_file_name}.csv"
    official_names = list(official_names_map.values())

    vmin = 0.2
    vmax = 0.95
    round_digits = 2
    col_names = official_names

    figsize = (len(col_names) * 4, 9)

    # create the dfs needed for heatmaps
    combined_table_top, metric_table_top = create_heatmap_dfs(metric_filename_top, std_filename_top, round_digits)
    # Create gap rows (filled with NaN for metrics and empty strings for annotations)
    combined_table_top, metric_table_top = insert_gap_row(combined_table_top, metric_table_top, gap_index=6)

    # Create a figure and gridspec for fine-tuned layout
    fig = plt.figure(figsize=figsize)
    heatmap = sn.heatmap(metric_table_top.astype(float),
                             annot=combined_table_top.values, fmt='s',
                             vmin=vmin, vmax=vmax,
                             annot_kws={"size": font_size},
                             cmap=cmap)

    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=font_size, rotation=0)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=font_size, rotation=0)
    heatmap.xaxis.tick_top()
    plt.title(title, fontdict={'fontsize': font_size + 10}, pad=15)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)

    dirpath = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\results_analysis\\st\\heatmaps"

    if image_path:
        brain_mask_img = plt.imread(os.path.join(dirpath, image_path))

        # Plotting the corresponding brain mask (ROI in the matrix computation)
        image_ax = fig.add_axes([0.885, 0.76, 0.12, 0.12])  # [left, bottom, width, height]
        image_ax.imshow(np.array(brain_mask_img))
        image_ax.axis('off')

    # Save the created plot
    Path(dirpath).mkdir(exist_ok=True)
    output_filepath = os.path.join(dirpath, f"{metric_file_name}.svg")

    plt.savefig(os.path.join(dirpath, output_filepath))
    print("Saved to: ", output_filepath)

def merged_heatmaps(metric_name,
                    image_path,
                    official_names_map,
                    font_size=20,
                    gap_size=0.175,
                    cmap="gist_rainbow"):
    metric_filename_top = f"metric_tables/metric_t1_to_t2_val_{metric_name}.csv"
    std_filename_top = f"std_tables/std_t1_to_t2_val_{metric_name}.csv"
    metric_filename_bottom = f"metric_tables/metric_t2_to_t1_val_{metric_name}.csv"
    std_filename_bottom = f"std_tables/std_t2_to_t1_val_{metric_name}.csv"

    official_names = list(official_names_map.values())

    if len(official_names) == 6:
        if "zoomed_ssim" in metric_filename_top:
            vmin = 0.5
            vmax = 0.7
            round_digits = 2
            col_names = official_names[-3:]
        elif "ssim" in metric_filename_top:
            vmin = 0.5
            vmax = 0.83
            round_digits = 2
            col_names = official_names
        elif "mse" in metric_filename_top:
            vmin = 7
            vmax = 40
            round_digits = 1
            col_names = official_names
        else:
            raise ValueError(f"Wrong file: {metric_filename_top}")


    figsize = (len(col_names) * 3, 20)
    # Loading the tables

    # create the dfs needed for heatmaps
    combined_table_top, metric_table_top = create_heatmap_dfs(metric_filename_top, std_filename_top, round_digits)
    combined_table_bottom, metric_table_bottom = create_heatmap_dfs(metric_filename_bottom, std_filename_bottom,
                                                                    round_digits)

    # # Create gap rows (filled with NaN for metrics and empty strings for annotations)
    combined_table_top, metric_table_top = insert_gap_row(combined_table_top, metric_table_top, gap_index=6)
    combined_table_bottom, metric_table_bottom = insert_gap_row(combined_table_bottom, metric_table_bottom, gap_index=6)

    # Create a figure and gridspec for fine-tuned layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=gap_size)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    heatmap_top = sn.heatmap(metric_table_top.astype(float),
                             annot=combined_table_top.values, fmt='s',
                             ax=ax1,
                             vmin=vmin, vmax=vmax,
                             annot_kws={"size": font_size},
                             cmap=cmap,
                             cbar=False)

    # Define the colorbar (to have just one)
    cbar_ax = fig.add_axes([0.93, 0.3, 0.02, 0.4])  # [left, bottom, width, height]

    heatmap_bottom = sn.heatmap(metric_table_bottom.astype(float),
                                annot=combined_table_bottom.values, fmt='s',
                                ax=ax2,
                                vmin=vmin, vmax=vmax,
                                annot_kws={"size": font_size},
                                cmap=cmap,
                                cbar_ax=cbar_ax)

    # Adjust the fontsize of rows and columns labels
    heatmap_top.set_xticklabels(heatmap_top.get_xticklabels(), fontsize=font_size, rotation=0)
    heatmap_top.set_yticklabels(heatmap_top.get_yticklabels(), fontsize=font_size, rotation=0)
    heatmap_top.xaxis.tick_top()
    ax1.set_title("$T_1$-weighted → $T_2$-weighted", fontdict={'fontsize': font_size + 10}, pad=15)

    heatmap_bottom.set_xticklabels(heatmap_bottom.get_xticklabels(), fontsize=font_size, rotation=0)
    heatmap_bottom.set_yticklabels(heatmap_bottom.get_yticklabels(), fontsize=font_size, rotation=0)
    heatmap_bottom.xaxis.tick_top()
    ax2.set_title("$T_2$-weighted → $T_1$-weighted", fontdict={'fontsize': font_size + 10}, pad=15)
    cbar_ax.yaxis.set_tick_params(labelsize=font_size)

    # Dynamically position vertical titles to the right of the heatmaps
    # for ax, title, y_center in zip([ax1, ax2], ["$T_1$ → $T_2$", "$T_2$ → $T_1$"], [0.75, 0.25]):
    #     box = ax.get_position()  # Get the position of the axis in figure coordinates
    #     fig.text(box.x1 + 0.001, (box.y0 + box.y1) / 2, title,
    #              va='center', ha='left', rotation=270, fontsize=font_size)

    # Line separating ST and FL results
    # ax1.hlines(6, 0, 1.0, color="black", linestyles='dashed', lw=4)
    # ax2.hlines(6, 0, 1.0, color="black", linestyles='dashed', lw=4)

    # divider_y = (ax1.get_position().y0 + ax2.get_position().y1) / 2
    divider_y = (10 * ax1.get_position().y0 / 16)
    fig.add_artist(plt.Line2D(
        [0.0, 0.9],  # Start and end points of the line (normalized x-coordinates)
        [divider_y, divider_y],  # y-coordinate for the line
        color="black",
        linestyle="dashed",
        linewidth=3,
        transform=fig.transFigure,
        clip_on=False
    ))

    # Load ROI image
    dirpath = "C:\\Users\\JanFiszer\\repos\\FLforMRItranslation\\results\\article\\tables"
    brain_mask_img = plt.imread(os.path.join(dirpath, image_path))

    # Plotting the corresponding brain mask (ROI in the matrix computation)
    image_ax = fig.add_axes([0.885, 0.76, 0.12, 0.12])  # [left, bottom, width, height]
    image_ax.imshow(np.array(brain_mask_img))
    image_ax.axis('off')

    # Save the created plot
    output_filepath = f"{metric_name}.svg"

    plt.savefig(os.path.join(dirpath, output_filepath))
    print("Saved to: ", output_filepath)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


if __name__ == "__main__":
    # metrics = ["masked_ssim", "masked_mse", "zoomed_ssim"]
    single_heatmap("val_gen_dice",
                   "Smoothed Dice",
                    official_names_map={"nonorm": "Raw", "minmax": "MinMax", "zscore": "Z-Score", "nyul": "Nyul", "fcm": "Fuzzy C-Mean", "whitestripe": "WhiteStripe"},
                    cmap="Spectral")