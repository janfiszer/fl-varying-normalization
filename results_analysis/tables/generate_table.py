import logging
import os

import pandas as pd
import pickle
from configs import config
from src.utils.files_operations import sort_by_substring_order


def get_evaluation_metrics(model_dir, ds_order, extract_test_set_fn=None, key_word=None):
    if key_word is None:
        key_word = "test_"
    if extract_test_set_fn is None:
        extract_test_set_fn = old_extract_test_set_name
    # finding files with the provided keyword
    eval_files = [filename for filename in os.listdir(model_dir) if key_word in filename]

    # sorting according to the order in official_names
    eval_files_sorted = sort_by_substring_order(eval_files, ds_order)

    print(eval_files_sorted)
    eval_metrics = {}

    for filename in eval_files_sorted:
        filepath = os.path.join(model_dir, filename)

        test_set_name = extract_test_set_fn(filename, key_word)
        with open(filepath, 'rb') as file:
            eval_metrics[test_set_name] = pickle.load(file)

    return eval_metrics


def old_extract_test_set_name(filename, key_word):
    return filename[len(key_word) + 1:-14]


def extract_3ddice_test_set_name(filename, key_word):
    return filename.split('_')[0]

def extract_for_exctracted_dir(filename, key_word):
    dataset_name = filename.split('_')[1]

    if dataset_name in {"metrics", "std"}:
        dataset_name = filename.split('_')[-3]
    if '-' in dataset_name:
        dataset_name = dataset_name.split('-')[0]

    return dataset_name


# def sort_by_substring_order(main_list, order_list):
#     # Create a key function to determine the sort order
#     def sort_key(string):
#         for index, substring in enumerate(order_list):
#             if substring in string:
#                 return index
#         return len(order_list)  # Put items without a match at the end
#
#     # Sort the main list using the key function
#     sorted_list = sorted(main_list, key=sort_key)
#
#     print(f"Sorted list: {sorted_list}, before {main_list}")
#     return sorted_list


def load_evals(main_dir, model_loading_order, ds_order, key_word, extra_dir=None):
    all_evals = []

    for model_dir in sort_by_substring_order(os.listdir(main_dir), model_loading_order):
        if "model" in model_dir:
            if extra_dir:
                model_path = os.path.join(main_dir, model_dir, extra_dir)
            else:
                model_path = os.path.join(main_dir, model_dir)
            print(f"Metric loaded from: {model_path}")
            all_evals.append(pd.DataFrame(get_evaluation_metrics(model_path, ds_order, key_word=key_word, extract_test_set_fn=extract_for_exctracted_dir)))

    return all_evals


def main():
    all_models_path_st = os.path.join("C:/Users/JanFiszer/repos/fl-varying-normalization/trained_models")

    # official_names_map = {"nonorm": "Raw", "minmax": "MinMax", "zscore": "Z-Score", "nyul": "Nyul", "fcm": "Fuzzy C-Mean", "whitestripe": "WhiteStripe"}
    order_st = list(config.OFFICIAL_NORMALIZATION_NAMES.keys())
    # extra_models = ["Centralized", "FedAvg", "FedAdam"]
    extra_models = ["Centralized", "FedAvg", "FedAdam", "FedBN", "FedMRI", "FedDelay", "Fed-BNDelay"]
    official_names = list(config.OFFICIAL_NORMALIZATION_NAMES.values()) + extra_models
    order_models = order_st + ["all"] + [name.lower() for name in extra_models[1:]]
    keyword = "std"
    st_evals = load_evals(all_models_path_st, order_models, order_st, keyword, extra_dir="3d_dice")

    for metric_name in ["val_gen_dice", "val_binarized_smoothed_dice", "val_binarized_jaccard_index"]:
        specified_metric_eval = {}

        for label, eval_metric in zip(official_names, st_evals):
            col_name = label
            try:
                specified_metric_eval[col_name] = eval_metric.loc[metric_name]
            except KeyError:  # an ugly way to deal with the fact that some evaluations result in metrics with a prefix "val_"
                specified_metric_eval[col_name] = eval_metric.loc[metric_name[len("val_"):]]

        df_specified_metric_eval = pd.DataFrame(specified_metric_eval)
        df_specified_metric_eval.index = list(config.OFFICIAL_NORMALIZATION_NAMES.values())

        # transposing the dataframe, to have the dataset as columns and models as rows
        df_specified_metric_eval = df_specified_metric_eval.T

        print("Data frame columns")
        print(df_specified_metric_eval.columns)

        # check if any rows has the same values
        if df_specified_metric_eval.duplicated().any():
            raise ValueError(f"Provided data has identical values duplicated are = {df_specified_metric_eval.duplicated()}. "
                             f"This is very improbable. Check the downloaded data.")
        df_specified_metric_eval.to_csv(f"results_analysis\\tables\\csv\\new_raw_model\{keyword}_{metric_name}.csv")


if __name__ == "__main__":
    main()
