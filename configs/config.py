import datetime
import logging
import os
from os import path
from configs.enums import *


"""
    This is the configuration file. 
    Not all variables are used in every run. It depends on how the server and clients are launched and which aggregation method is used.
"""

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

# Variable set to true for testing locally
# It affects i.a. filepaths construction, server address.
# Also determines if multiple workers are used by DataLoader (see PyTorch dataloader)
LOCAL = False

# DEEP LEARNING PARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.0
NUM_WORKERS = 8
# model parameters 
NORMALIZATION = LayerNormalizationType.GN  # tested and group normalization works the best
N_GROUP_NORM = 32

# saving and logging while training
USE_WANDB = True
BATCH_PRINT_FREQ = 10  # number of batch after each the training parameters (metrics, loss) is printed
SAVING_FREQUENCY = 10  # how often (round-wise) the model is saved
CLIENT_SAVING_FREQ = 10  # how often (round-wise) the model is saved for client
PLOT_BATCH_WITH_METRICS = False
PLOT_EACH_EPOCH = True

USED_MODALITIES = ["t1", "t2", "flair"]
MASK_DIR = "mask"

# METRIC PARAMETERS
METRICS = ["loss", "torch_multi_class_gen_dice", "torch_multi_per_class_gen_dice"]
# Dice
NUM_CLASSES = 3
INCLUDE_BACKGROUND = True  # whether to include background as a separate class
DICE_WEIGHT_TYPE = 'linear'  # 'linear' or 'square'

# Federated learning
# USED ONLY: when the server and clients are started separately
# port address, for communication
PORT = "8084"
# training parameters
CLIENT_TYPE = ClientTypes.FED_BN
AGGREGATION_METHOD = AggregationMethods.FED_AVG 

# federated learning parameters
N_ROUNDS = 32
MIN_FIT_CLIENTS = MIN_AVAILABLE_CLIENTS = 6
FRACTION_FIT = 1.0

# Client site
N_EPOCHS_CLIENT = 2


# SPECIALIZED METHOD
# FedOpt
TAU = 1e-3
# FedProx
PROXIMAL_MU = 0.001
STRAGGLERS = 0.5
# FedAvgM
MOMENTUM = 0.9
# FedDelay
FEDDELAY_ROUND = 30

# centralized train
N_EPOCHS_CENTRALIZED = 32

# file names and paths
PROJECT_NAME = "fl-varying-normalization"
NODE_FILENAME = "SERVERNODE.txt"
NODE_SERVER_DIRPATH = "comunication/server_nodes"
# train, test and validation directories names
TRAIN_DIR_NAME = "train"
TEST_DIR_NAME = "test"
VAL_DIR_NAME = "validation"

# naming 
OFFICIAL_MODALITIES_NAMES = {short: official for short, official in zip(USED_MODALITIES, ["$T_1$-weighted", "$T_2$-weighted", "FLAIR"])}
OFFICIAL_NORMALIZATION_NAMES = {"nonorm": "Raw", "minmax": "MinMax", "zscore": "Z-Score", "nyul": "Nyul", "fcm": "Fuzzy C-Mean", "whitestripe": "WhiteStripe"}
NORMALIZATION_ORDER = list(OFFICIAL_NORMALIZATION_NAMES.keys())

# directories
MODALITIES_AND_PATHS_FROM_LOCAL_DIR = {"t1": "*T1.nii.gz",
                                       "t2": "*T2.nii.gz",
                                       "flair": "*FLAIR.nii.gz",
                                       "mask": "*tumor_segmentation.nii.gz"
                                       }
if LOCAL:
    DATA_ROOT_DIR = path.join(path.expanduser("~"), "data")
else:
    DATA_ROOT_DIR = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/varying-normalization"

TRAINED_MODELS_DIR = path.join(DATA_ROOT_DIR, "trained_models/multi_class")

now = datetime.datetime.now()
# CENTRALIZED_DIR = f"{DATA_ROOT_DIR}/trained_models/model-mgh-centralized-{LOSS_TYPE.name}-ep{N_EPOCHS_CENTRALIZED}-{TRANSLATION[0].name}{TRANSLATION[1].name}-lr{LEARNING_RATE}-{now.date()}-{now.hour}h"
_REPRESENTATIVE_WORD = CLIENT_TYPE if CLIENT_TYPE == ClientTypes.FED_BN or CLIENT_TYPE == ClientTypes.FED_MRI else AGGREGATION_METHOD
# TRAINED_MODEL_SERVER_DIR = f"{DATA_ROOT_DIR}/trained_models/model-{_REPRESENTATIVE_WORD.name}-{LOSS_TYPE.name}-{TRANSLATION[0].name}{TRANSLATION[1].name}-lr{LEARNING_RATE}-rd{N_ROUNDS}-ep{N_EPOCHS_CLIENT}-{now.date()}"
