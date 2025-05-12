import os

import pandas as pd
import pickle


def get_evaluation_metrics(model_dir, ds_order, key_word=None):
    if key_word is None:
        key_word = "test_"

    # finding files with the provided keyword
    eval_files = [filename for filename in os.listdir(model_dir) if key_word in filename]

    # sorting according to the order in official_names
    eval_files_sorted = sort_by_substring_order(eval_files, ds_order)

    print(eval_files_sorted)
    eval_metrics = {}

    for filename in eval_files_sorted:
        filepath = os.path.join(model_dir, filename)

        test_set_name = filename[len(key_word) + 1:-14]
        with open(filepath, 'rb') as file:
            eval_metrics[test_set_name] = pickle.load(file)

    return eval_metrics


def sort_by_substring_order(main_list, order_list):
    # Create a key function to determine the sort order
    def sort_key(string):
        for index, substring in enumerate(order_list):
            if substring in string:
                return index
        return len(order_list)  # Put items without a match at the end

    # Sort the main list using the key function
    sorted_list = sorted(main_list, key=sort_key)
    return sorted_list


def load_evals(main_dir, model_loading_order, ds_order, key_word):
    all_evals = []

    for model_dir in sort_by_substring_order(os.listdir(main_dir), model_loading_order):
        if "model" in model_dir:
            model_path = os.path.join(main_dir, model_dir)
            print(f"Metric loaded from: {model_path}")
            all_evals.append(pd.DataFrame(get_evaluation_metrics(model_path, ds_order, key_word=key_word)))

    return all_evals


def main():
    all_models_path_st = os.path.join("C:/Users/JanFiszer/repos/fl-varying-normalization/trained_models")

    official_names_map = {"nonorm": "Raw", "minmax": "MinMax", "zscore": "Z-Score", "nyul": "Nyul", "fcm": "Fuzzy C-Mean", "whitestripe": "WhiteStripe"}
    order_st = official_names_map.keys()
    official_names = list(official_names_map.values()) + ["Centralized", "FedAvg"]

    keyword = "std"
    st_evals = load_evals(all_models_path_st, order_st, order_st, keyword)

    for metric_name in ["val_gen_dice", "val_binarized_smoothed_dice", "val_binarized_jaccard_index"]:
        specified_metric_eval = {}

        for label, eval_metric in zip(official_names, st_evals):
            col_name = label
            specified_metric_eval[col_name] = eval_metric.loc[metric_name]

        df_specified_metric_eval = pd.DataFrame(specified_metric_eval)
        df_specified_metric_eval.index = list(official_names_map.values())

        # transposing the dataframe, to have the dataset as columns and models as rows
        df_specified_metric_eval = df_specified_metric_eval.T

        print("Data frame columns")
        print(df_specified_metric_eval.columns)

        # check if any rows has the same values
        if df_specified_metric_eval.duplicated().any():
            raise ValueError(f"Provided data has identical values duplicated are = {df_specified_metric_eval.duplicated()}. "
                             f"This is very improbable. Check the downloaded data.")
        df_specified_metric_eval.to_csv(f"tables\\{keyword}_{metric_name}.csv")


if __name__ == "__main__":
    main()
