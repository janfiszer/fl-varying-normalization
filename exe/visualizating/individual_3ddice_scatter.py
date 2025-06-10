import os
import pickle
import torch
from configs import config
from src.utils.visualization import plot_dice_scores_combined


def load_dice_scores_from_pickles(pickle_files, metrics_name):
    """
    Load DICE scores from multiple pickle files.

    Args:
        pickle_files: List of tuples (file_path, model_name) or dict {model_name: file_path}

    Returns:
        dict: {model_name: {patient_id: dice_score}}
    """
    all_scores = {}

    if isinstance(pickle_files, dict):
        # If input is a dictionary
        for model_name in config.PRESENTED_MODEL_DIRs:
            try:
                file_path = pickle_files[model_name]
                with open(file_path, 'rb') as f:
                    scores = pickle.load(f)
                    # Convert tensors to float values
                    processed_scores = {}
                    for patient_id, score in scores[metrics_name].items():
                        if isinstance(score, torch.Tensor):
                            processed_scores[patient_id] = score.item()
                        else:
                            processed_scores[patient_id] = float(score)
                    all_scores[model_name] = processed_scores
                    print(f"Loaded {len(processed_scores)} scores for {model_name}")
            except Exception as e:
                print(f"Error loading {file_path} for {model_name}: {e}")
    else:
        raise NotImplementedError
    #     # If input is a list of tuples
    #     for file_path, model_name in pickle_files:
    #         try:
    #             with open(file_path, 'rb') as f:
    #                 scores = pickle.load(f)
    #                 # Convert tensors to float values
    #                 processed_scores = {}
    #                 for patient_id, score in scores.items():
    #                     if isinstance(score, torch.Tensor):
    #                         processed_scores[patient_id] = score.item()
    #                     else:
    #                         processed_scores[patient_id] = float(score)
    #                 all_scores[model_name] = processed_scores
    #                 print(f"Loaded {len(processed_scores)} scores for {model_name}")
    #         except Exception as e:
    #             print(f"Error loading {file_path} for {model_name}: {e}")

    return all_scores


def get_pickle_files(dir_path, concerned_dataset, dice3d_dir_name):
    found_pickle_files = {}

    for inner_dir_name in os.listdir(dir_path):
        if "model" in inner_dir_name:
            found_pickle_files[inner_dir_name] = os.path.join(trained_models_dir, inner_dir_name, dice3d_dir_name, f"{concerned_dataset.lower()}.pkl")

    return found_pickle_files


if __name__ == "__main__":
    concerned_dataset = "FCM"
    trained_models_dir = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\trained_models"
    inner_dataset = "individual_3d_dice"
    sample_pickle_files = {
        'ST Raw': os.path.join(trained_models_dir, 'model-nonorm-ep16-lr0.001-GN-2025-04-30-13h', inner_dataset,
                               f"{concerned_dataset.lower()}.pkl"),
        'ST MinMax': os.path.join(trained_models_dir, 'model-minmax-ep16-lr0.001-GN-2025-04-30-13h', inner_dataset,
                                  f"{concerned_dataset.lower()}.pkl"),
        'ST WhiteStipe': os.path.join(trained_models_dir, 'model-whitestripe-ep16-lr0.001-GN-2025-04-30-13h', inner_dataset,
                                      f"{concerned_dataset.lower()}.pkl"),
        'FedAdam': os.path.join(trained_models_dir, 'model-nonorm-ep16-lr0.001-GN-2025-04-30-13h', inner_dataset,
                                f"{concerned_dataset.lower()}.pkl"),
    }

    all_datasets = config.OFFICIAL_NORMALIZATION_NAMES

    for dataset, dataset_official_name in all_datasets.items():
        print(f"Normalization method: {dataset_official_name}")
        pickle_files = get_pickle_files(trained_models_dir, dataset, inner_dataset)

        # Load scores from pickle files
        all_scores = load_dice_scores_from_pickles(pickle_files, 'gen_dice')

        # Create visualization
        if all_scores:
            # Single combined plot with all models
            plot_dice_scores_combined(all_scores, title=f"DICE Scores for individual patient normalized by {dataset_official_name}", patient_plot_zone_width=40)
        else:
            print("No data loaded. Please check your pickle file paths.")
