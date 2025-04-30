#!/bin/bash

# SYSTEM CONFIGURATION
export LOCAL=false

# DEEP LEARNING PARAMETERS
# export LOSS_TYPE="MSE_DSSIM"
export BATCH_SIZE=32
export LEARNING_RATE=0.001
export DROPOUT_RATE=0.3
export N_EPOCHS_CENTRALIZED=16
# export NUM_WORKERS=8
# export NORMALIZATION="GN"
# export N_GROUP_NORM=32

# # DATA PARAMETERS
# export METRICS="loss,gen_dice,binarized_smoothed_dice,binarized_jaccard_index"
# export USED_MODALITIES="t1,t2,flair"
# export MASK_DIR="mask"

# FEDERATED LEARNING PARAMETERS
export PORT="8084"
export CLIENT_TYPE="FED_BN"
export AGGREGATION_METHOD="FED_AVG"
# export N_ROUNDS=32
# export MIN_FIT_CLIENTS=4
# export MIN_AVAILABLE_CLIENTS=4
# export FRACTION_FIT=1.0
# export N_EPOCHS_CLIENT=4

# CENTRALIZED TRAINING PARAMETERS

# DATA PATHS
export DATA_ROOT_DIR="/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/varying-normalization/data"
export DATA_DIR="${DATA_ROOT_DIR}/zscore"
# MODALITIES CONFIG (comma-separated key:value pairs)
# export MODALITIES_CONFIG="t1:*T1.nii.gz,t2:*T2.nii.gz,flair:*FLAIR.nii.gz,mask:*tumor_segmentation.nii.gz"

ARGS=""
ARGS+=" --batch-size ${BATCH_SIZE}"
ARGS+=" --learning-rate ${LEARNING_RATE}"
ARGS+=" --dropout-rate ${DROPOUT_RATE}"
ARGS+=" --n-epochs-centralized ${N_EPOCHS_CENTRALIZED}"
ARGS+=" --data-dir ${DATA_DIR}"
# ARGS+=" --normalization ${NORMALIZATION}"
# ARGS+=" --n-group-norm ${N_GROUP_NORM}"
# ARGS+=" --aggregation-method ${AGGREGATION_METHOD}"

run_with_args() {
    local script_name=$1
    echo "Running: python -m $script_name $ARGS"
    $PLG_GROUPS_STORAGE/plggflmri/new_conda/fl/bin/python -u -m $script_name $ARGS
}
