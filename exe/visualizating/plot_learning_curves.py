import os

import pickle

import matplotlib.pyplot as plt
from src.utils import visualization
from configs import config, enums
from torch.utils.data import DataLoader
from configs.enums import LayerNormalizationType
from matplotlib import rcParams


official_names_map = {"nonorm": "No Normalization",  "minmax": "MinMax",  "zscore": "Z-Score", "whitestripe": "WhiteStripe", "nyul": "Nyul", "fcm": "Fuzzy C-Mean"}
official_names = list(official_names_map.values())

trained_models_dir = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\trained_models\\st"
model_dirs = ["model-zscore-ep16-lr0.001-GN-2025-04-30-11%3A06-bilinear",
              "model-zscore-ep16-lr0.001-GN-2025-04-29-13h"]


for model_dir in model_dirs:
    model_dir_path = os.path.join(trained_models_dir, model_dir)
    with open(f"{model_dir_path}/history.pkl", 'rb') as f:
        history = pickle.load(f)
    visualization.plot_history(history, os.path.join(model_dir_path, "learning_curves.png"))
