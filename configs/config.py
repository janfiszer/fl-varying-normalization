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
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Variable set to true for testing locally
# It affects i.a. filepaths construction, server address.
# Also determines if multiple workers are used by DataLoader (see PyTorch dataloader)
LOCAL = False

# DEEP LEARNING PARAMETERS
LOSS_TYPE = LossFunctions.MSE_DSSIM
NUM_WORKERS = 8

# model parameters
NORMALIZATION = LayerNormalizationType.GN
N_GROUP_NORM = 32

# saving and logging while training
USE_WANDB = False
BATCH_PRINT_FREQ = 10  # number of batch after each the training parameters (metrics, loss) is printed
SAVING_FREQUENCY = 10  # how often (round-wise) the model is saved
CLIENT_SAVING_FREQ = 10  # how often (round-wise) the model is saved for client
PLOT_BATCH_WITH_METRICS = False
PLOT_EACH_EPOCH = True

METRICS = ["loss", "gen_dice", "binarized_smoothed_dice", "binarized_jaccard_index"]
USED_MODALITIES = ["t1", "t2", "flair"]
MASK_DIR = "mask"

# Federated learning
# USED ONLY: when the server and clients are started separately
# port address, for communication
PORT = "8084"
# training parameters
CLIENT_TYPE = ClientTypes.FED_BN
AGGREGATION_METHOD = AggregationMethods.FED_AVG 

# federated learning parameters
N_ROUNDS = 32
MIN_FIT_CLIENTS = MIN_AVAILABLE_CLIENTS = 4
FRACTION_FIT = 1.0

# Client site
N_EPOCHS_CLIENT = 4


# SPECIALIZED METHOD
# FedOpt
TAU = 1e-3
# FedProx
PROXIMAL_MU = 0.001
STRAGGLERS = 0.5
# FedAvgM
MOMENTUM = 0.9

# file names and paths
PROJECT_NAME = "fl-varying-normalization"
NODE_FILENAME = "SERVERNODE.txt"

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

now = datetime.datetime.now()

# Defind in the config.sh
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
N_EPOCHS_CENTRALIZED = 16
