import logging
import os
import os.path
import pickle
from pathlib import Path
import random
from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from configs import enums, config as global_config
from src.deep_learning import models
from src.deep_learning.datasets import SegmentationDataset2DSlices


class ClassicClient(fl.client.NumPyClient):
    """
        Overriding all the methods that NumPyClient requires.
    """
    def __init__(self, client_id, model: models.UNet, optimizer, data_dir, model_dir, loss_name="val_loss", save_best_model_filename=None):
        """
            Constructor

            Parameters
            ----------
            client_id:
                Client representative, string. Usually set to the name of the dataset.
            model:
                Client model which he will update during local training
            optimizer:
                Model's optimizer, used as the loss function
            data_dir:
                Full path to the data directory, which client trains on. Requires to have inside (named exactly):
                    - train
                    - test 
                    - validation
            model_dir:
                A directory, inside which a directory based on client_id and client type is created. 
                There the client saves its model and its potential test plot, 
        """
        self.client_id = client_id
        self.model = model
        self.loss_name = loss_name
        self.save_best_model_filename = save_best_model_filename
        self.current_best_loss = float('inf')
        self.optimizer = optimizer

        self.train_loader, self.test_loader, self.val_loader = load_data(data_dir,
                                                                         batch_size=global_config.BATCH_SIZE,
                                                                         with_num_workers=not global_config.LOCAL)


        self.history = {f"val_{metric_name}": [] for metric_name in global_config.METRICS}
        self.client_dir = os.path.join(model_dir,
                                       f"{self.__repr__()}_client_{self.client_id}")

        # creating the client directory
        Path(self.client_dir).mkdir()
        logging.info(f"Client {client_id} with data from directory: {data_dir}: INITIALIZED\n")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: NDArrays, config):
        self.set_parameters(parameters)

        current_round = config["current_round"]
        logging.info(f"ROUND {current_round}")

        if global_config.LOCAL:
            plots_dir = None
        else:
            plots_dir = f"{self.client_dir}/rd-{current_round}_training_plots"

        history = self.model.perform_train(self.train_loader,
                                           self.optimizer,
                                           model_dir=self.client_dir,
                                           validationloader=self.val_loader,
                                           epochs=global_config.N_EPOCHS_CLIENT,
                                        #    plots_dir=plots_dir
                                           )

        logging.info(f"END OF CLIENT TRAINING\n")

        val_metric_names = [f"val_{metric}" for metric in global_config.METRICS]

        # only validation metrics from the client (ensured by 'val_' suffix)
        avg_val_metric = {
            metric_name: sum([metric_value for metric_value in history[metric_name]]) / len(history[metric_name])
            for metric_name in val_metric_names}

        avg_val_metric["client_id"] = self.client_id  # client_id to keep truck of the loss properly e.g. FedCostW

        return self.get_parameters(config=config), len(self.train_loader.dataset), avg_val_metric

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)

        metrics = self._evaluate(current_round=config["current_round"])
        metric_without_loss = {k: v for k, v in metrics.items() if k != self.loss_name}

        if metrics[self.loss_name] < self.current_best_loss:
            self.model.save(self.client_dir, self.save_best_model_filename)
            logging.info(f"Model from round {config['current_round']} has the best loss value so far: {metrics[self.loss_name]:.3f}. Current best is {self.current_best_loss:.3f}")
            self.current_best_loss = metrics[self.loss_name]

        return metrics[self.loss_name], len(self.test_loader.dataset), metric_without_loss

    def _evaluate(self, current_round: int):

        logging.info(f"CLIENT {self.client_id} ROUND {current_round} TESTING...")

        if global_config.LOCAL:
            plots_path = None
            plot_filename = None
        else:
            plots_path = f"{self.client_dir}/test_plots"
            plot_filename = f"round-{current_round}"

        metrics = self.model.evaluate(self.test_loader,
                                      plots_path=plots_path,
                                      plot_last_batch_each_epoch=True,
                                      epoch_number=current_round  # TODO: instead of epoch number back to filename?
                                      )

        logging.info(f"END OF CLIENT TESTING\n\n")

        # adding to the history
        for metric_name, metric_value in metrics.items():
            self.history[metric_name].append(metric_value)

        if current_round % global_config.CLIENT_SAVING_FREQ == 0:
            self.model.save(self.client_dir, f"model-rd{current_round}")

        # saving model and history if it is the last round
        if current_round == global_config.N_ROUNDS:
            self.model.save(self.client_dir)

            with open(f"{self.client_dir}/history.pkl", 'wb') as file:
                pickle.dump(self.history, file)

        return metrics

    def __repr__(self):
        return "FedAvg"


class FedBNClient(ClassicClient):
    """Changes only the parameters operation (set and get) skipping the normalization layers"""

    # NOT NEEDED (SAME US FEDMRI) 
    # def get_parameters(self, config) -> NDArrays:
    #     return [val.cpu().numpy() for name, val in self.model.state_dict().items()]
    #     # return [val.cpu().numpy() for layer_name, val in self.model.state_dict().items()
    #             # if "norm" not in layer_name]

    def set_parameters(self, parameters):
        self.model.train()

        old_state_dict = self.model.state_dict()

        # Excluding parameters of BN layers when using FedBN
        layer_names = {index: layer_name for index, layer_name in enumerate(old_state_dict.keys())
                       if "norm" not in layer_name}

        selected_parameters = [parameters[i] for i in layer_names.keys()]
        param_dict = zip(layer_names.values(), selected_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def __repr__(self):
        return f"FedBN(batch_norm={global_config.NORMALIZATION})"


class FedMRIClient(ClassicClient):
    "Changes only the parameters operation (set and get) skipping the decoder part. Only encoder in global"
    def set_parameters(self, parameters: NDArrays):
        self.model.train()

        old_state_dict = self.model.state_dict()

        layer_names = {index: layer_name for index, layer_name in enumerate(old_state_dict.keys())
                       if "down" in layer_name or "inc" in layer_name}

        selected_parameters = [parameters[i] for i in layer_names.keys()]
        param_dict = zip(layer_names.values(), selected_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def __repr__(self):
        return f"FedMRI()"


def client_from_string(client_id, unet: models.UNet, optimizer, data_dir: str, client_type_name):
    """
        Returns instance of a class basing on the given string. Requires the model (Pytorch net) and optimizer.
        Client ID is a string.
    """
    drd = global_config.DATA_ROOT_DIR
    lr = global_config.LEARNING_RATE
    rd = global_config.N_ROUNDS
    ec = global_config.N_EPOCHS_CLIENT
    d = global_config.now.date()
    h = global_config.now.hour

    model_dir = f"{drd}/trained_models/model-{client_type_name}-lr{lr}-rd{rd}-ep{ec}-{d}"
    
    logging.info(f"Client {client_id} has directory: {model_dir}")

    if client_type_name in ["fedbn"]:
        return FedBNClient(client_id, unet, optimizer, data_dir, model_dir, save_best_model_filename="best_model")
    elif client_type_name in ["fedmri"]:
        return FedMRIClient(client_id, unet, optimizer, data_dir, model_dir, save_best_model_filename="best_model")
    elif  client_type_name in ["fedavg"]:
        return ClassicClient(client_id, unet, optimizer, data_dir, model_dir)
    
    else:
        raise ValueError(f"Given client type ('{client_type_name}') name is invalid.")


def load_data(data_dir, batch_size, with_num_workers=True):
    """
        Function returning training, test and validation loader based on same as named directories in the given directory (data_dir)
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    val_dir = os.path.join(data_dir, "validation")

    args = (global_config.USED_MODALITIES, global_config.MASK_DIR, True)

    trainset = SegmentationDataset2DSlices([train_dir],  *args)
    testset = SegmentationDataset2DSlices([test_dir], *args)
    validationset = SegmentationDataset2DSlices([val_dir], *args)

    if with_num_workers:
        train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=global_config.NUM_WORKERS,
                                  pin_memory=True, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=global_config.NUM_WORKERS,
                                 pin_memory=True, shuffle=True)
        val_loader = DataLoader(validationset, batch_size=batch_size, num_workers=global_config.NUM_WORKERS,
                                pin_memory=True, shuffle=True)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader
