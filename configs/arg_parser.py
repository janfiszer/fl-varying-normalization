import argparse
import os
from os import path
import logging
from configs.enums import *
from configs import config as deafault_config


def parse_args():
    """Parse command line arguments and create configuration."""
    parser = argparse.ArgumentParser(description='Neural Network Training Configuration')
    
    # DEEP LEARNING PARAMETERS
    parser.add_argument('--batch-size', type=int, default=deafault_config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=deafault_config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--dropout-rate', type=float, default=deafault_config.DROPOUT_RATE,
                        help='Dropout rate')
    parser.add_argument('--n-epochs-centralized', type=int, default=deafault_config.N_EPOCHS_CENTRALIZED,
                    help='Number of epochs')
    parser.add_argument('--data-dir', type=str, default="dupa", # TODO: dupa
                    help='Directory where data is stored')

    args = parser.parse_args()
    
    # Convert string arguments to appropriate types
    config = {}
    
    # Process simple parameters
    config['BATCH_SIZE'] = args.batch_size
    config['LEARNING_RATE'] = args.learning_rate
    config['DROPOUT_RATE'] = args.dropout_rate
    config['N_EPOCHS_CENTRALIZED'] = args.n_epochs_centralized
    config["DATA_DIR"] = args.data_dir
    # Print configuration
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    
    return config