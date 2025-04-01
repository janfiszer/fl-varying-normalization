import datetime
from os import path
from configs.enums import *


"""
    This is the configuration file. 
    Not all variables are used in every run. It depends on how the server and clients are launched and which aggregation method is used.
"""


# Variable set to true for testing locally
# It affects i.a. filepaths construction, server address.
# Also determines if multiple workers are used by DataLoader (see PyTorch dataloader)
LOCAL = True
NODE_FILENAME = "SERVERNODE.txt"

# saving and logging
USE_WANDB = False
BATCH_PRINT_FREQ = 10  # number of batch after each the training parameters (metrics, loss) is printed
SAVING_FREQUENCY = 10  # how often (round-wise) the model is saved
CLIENT_SAVING_FREQ = 10  # how often (round-wise) the model is saved for client

# model parameters
NORMALIZATION = NormalizationType.BN
N_GROUP_NORM = 32


# client parameters
METRICS = ["loss",  "mse", "relative_error", "ssim", "zoomed_ssim"]

# METRICS = ["loss", "ssim", "pnsr", "mse", "masked_mse", "relative_error"]
# METRICS = ["loss", "ssim", "masked_ssim", "pnsr", "mse", "masked_mse", "relative_error"]
N_EPOCHS_CLIENT = 4

TRANSLATION = (ImageModality.T1, ImageModality.T1)
LOSS_TYPE = LossFunctions.MSE_DSSIM
BATCH_SIZE = 32
IMAGE_SIZE = (240, 240)
LEARNING_RATE = 0.001
NUM_WORKERS = 8


# USED ONLY: when the server and clients are started separately
# port address
PORT = "8084"
# training parameters
CLIENT_TYPE = ClientTypes.FED_BN
AGGREGATION_METHOD = AggregationMethods.FED_AVG 

# federated learning parameters
N_ROUNDS = 32
MIN_FIT_CLIENTS = MIN_AVAILABLE_CLIENTS = 4
FRACTION_FIT = 1.0

# SPECIALIZED METHOD
# FedOpt
TAU = 1e-3
# FedProx
PROXIMAL_MU = 0.001
STRAGGLERS = 0.5
# FedAvgM
MOMENTUM = 0.9

# centralized train
N_EPOCHS_CENTRALIZED = 10

# directories
if LOCAL:
    DATA_ROOT_DIR = path.join(path.expanduser("~"), "data")
else:
    DATA_ROOT_DIR = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL"

EVAL_DATA_DIRS = [path.join(DATA_ROOT_DIR, "lgg_26", "test"),
                  path.join(DATA_ROOT_DIR, "hgg_26", "test"),
                  path.join(DATA_ROOT_DIR, "wu_minn_26", "test"),
                  path.join(DATA_ROOT_DIR, "hcp_mgh_masks", "test"),
                  path.join(DATA_ROOT_DIR, "oasis_26", "test")]

now = datetime.datetime.now()
# CENTRALIZED_DIR = f"{DATA_ROOT_DIR}/trained_models/model-mgh-centralized-{LOSS_TYPE.name}-ep{N_EPOCHS_CENTRALIZED}-{TRANSLATION[0].name}{TRANSLATION[1].name}-lr{LEARNING_RATE}-{now.date()}-{now.hour}h"
_REPRESENTATIVE_WORD = CLIENT_TYPE if CLIENT_TYPE == ClientTypes.FED_BN or CLIENT_TYPE == ClientTypes.FED_MRI else AGGREGATION_METHOD
TRAINED_MODEL_SERVER_DIR = f"{DATA_ROOT_DIR}/trained_models/model-{_REPRESENTATIVE_WORD.name}-{LOSS_TYPE.name}-{TRANSLATION[0].name}{TRANSLATION[1].name}-lr{LEARNING_RATE}-rd{N_ROUNDS}-ep{N_EPOCHS_CLIENT}-{now.date()}"
