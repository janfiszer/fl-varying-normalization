import time
import pickle
import os
from shutil import copy2
from pathlib import Path
import torch
import logging
import numpy as np

from functools import reduce
from torch.utils.data import DataLoader

from src.deep_learning import datasets, models
from src.utils import files_operations
from configs import config

import flwr as fl
from flwr.server.criterion import ClientProxy
from flwr.common import Scalar, FitRes, Parameters, logger, Metrics, NDArrays, parameters_to_ndarrays, \
    ndarrays_to_parameters, NDArray
from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAdam, FedAvg, FedYogi, FedProx, FedAdagrad, FedAvgM, aggregate

from typing import List, Tuple, Dict, Union, Optional, Type, Callable
from collections import OrderedDict


def create_dynamic_strategy(StrategyClass: Type[Strategy], model: models.UNet, model_dir, *args, **kwargs):
    """ A function that returns a strategy class instance that will return  
    """
    class SavingModelStrategy(StrategyClass):
        def __init__(self):
            initial_parameters = [val.cpu().numpy() for val in model.state_dict().values()]
            super().__init__(initial_parameters=ndarrays_to_parameters(initial_parameters), *args, **kwargs)
            self.model = model
            self.model_dir = model_dir
            self.aggregation_times = []
            self.best_loss = float('inf')

            Path(self.model_dir).mkdir(exist_ok=True)  # creating directory before to don't get warnings
            copy2("./configs/config.py", f"{self.model_dir}/config.py")

        def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # start counting time
            start = time.time()

            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                        
            # printing and saving aggregation times
            aggregation_time = time.time() - start
            self.aggregation_times.append(aggregation_time)
            logging.info(f"\n{self.__str__()} aggregation time: {aggregation_time}\n")

            # computing average loss
            loss_values = [fit_res.metrics["val_loss"] for _, fit_res in results]
            current_avg_loss = sum(loss_values)/len(loss_values)

            # saving model
            self.save_model_conditionally(aggregated_parameters, server_round, current_avg_loss=current_avg_loss)

            return aggregated_parameters, aggregated_metrics

        def save_model_conditionally(self, aggregated_parameters, server_round, save_last_round=True, save_intervals=config.SAVING_FREQUENCY, current_avg_loss=None):
            # saving in intervals
            if save_intervals:
                if server_round % config.SAVING_FREQUENCY == 1:
                    save_aggregated_model(self.model, aggregated_parameters, self.model_dir, server_round)

            # saving the best model
            if current_avg_loss:
                if current_avg_loss < self.best_loss:
                    logging.info(f"Best model with loss {current_avg_loss:.3f}<{self.best_loss:.3f}")
                    save_aggregated_model(self.model, aggregated_parameters, self.model_dir, server_round, best_model=True)
                    self.best_loss = current_avg_loss
                else:
                    logging.info(f"Best model with loss {current_avg_loss:.3f}>{self.best_loss:.3f}")

            # saving in the last round
            if save_last_round:
                if server_round == config.N_ROUNDS:
                    # model
                    save_aggregated_model(self.model, aggregated_parameters, self.model_dir, server_round)
                    # aggregation times
                    with open(f"{self.model_dir}/aggregation_times.pkl", "wb") as file:
                        pickle.dump(self.aggregation_times, file)

    return SavingModelStrategy()


def save_aggregated_model(model: models.UNet, aggregated_parameters, model_dir, server_round: int, best_model=False):
    """
        Takes aggregated parameters and saves them to the model_dir with name describing the current round.
    """
    aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

    params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    # saving the model with an appropriate name
    model_name = "best.pth" if best_model else f"round{server_round}.pth"
    torch.save(state_dict, os.path.join(model_dir, model_name))

    # it could have been done using
    # model.load_state_dict(state_dict)
    # model.save(model_dir, filename=f"round{server_round}.pth")

    logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {model_dir}")

def strategy_from_string(model, strategy_name, evaluate_fn=None):
    """
        Returns client object. Basing on the strategy name different aggregation methods are chosen.
        Asignes appropriate parameters from config if they are needed by the aggreagation method.
        model_dir is constructed basing on the config_train.
    """

    # the directory includes the strategy name
    # so when it is initialized by the string it is created here
    # by default it takes the name TRAINED_MODEL_SERVER_DIR
    drd = config.DATA_ROOT_DIR
    lr = config.LEARNING_RATE
    rd = config.N_ROUNDS
    ec = config.N_EPOCHS_CLIENT
    d = config.now.date()
    h = config.now.hour

    model_dir = f"{drd}/trained_models/model-{strategy_name}-lr{lr}-rd{rd}-ep{ec}-{d}"

    ## FOR NOW CREATION IN THE STRATEGY CONSTRUCTOR
    # for optimal from_config usage created in the strategy constructor
    # files_operations.try_create_dir(model_dir)
    # copy2("./configs/config_train.py", f"{model_dir}/config.py")

    kwargs = {
        "min_fit_clients": config.MIN_FIT_CLIENTS,
        "min_available_clients": config.MIN_AVAILABLE_CLIENTS,
        "fraction_fit": config.FRACTION_FIT,
        "on_fit_config_fn": get_on_fit_config(),
        "evaluate_fn": evaluate_fn,
        "on_evaluate_config_fn": get_on_eval_config()
    }

    if strategy_name in ["fedadam", "fedmix", "fedbadam"]:
        strategy_class = FedAdam
        kwargs["tau"] = config.TAU

    elif strategy_name in ["fedavg", "fedbn"]:
        strategy_class = FedAvg
    else:
        raise ValueError(f"Wrong starategy name: {strategy_name}")

    return create_dynamic_strategy(strategy_class, model, model_dir, **kwargs)


def strategy_from_config(model, evaluate_fn=None):
    kwargs = {
        "min_fit_clients": config.MIN_FIT_CLIENTS,
        "min_available_clients": config.MIN_AVAILABLE_CLIENTS,
        "fraction_fit": config.FRACTION_FIT,
        "on_fit_config_fn": get_on_fit_config(),
        "evaluate_fn": evaluate_fn,
        "on_evaluate_config_fn": get_on_eval_config()
    }

    if config.AGGREGATION_METHOD == config.AggregationMethods.FED_ADAM:
        strategy_class = FedAdam
        kwargs["tau"] = config.TAU
    else:  # FedAvg or FedBN
        strategy_class = FedAvg

    return create_dynamic_strategy(strategy_class, model, **kwargs)


# FUNCTIONS
# used by the strategy to during fit and evaluate
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # val_metric_names = [f"val_{metric}" for metric in config.METRICS]
    results = {f"val_metric_name": 0.0 for metric_name in config.METRICS}

    for num_examples, m in metrics:
        for metric_name in results.keys():
            results[metric_name] += num_examples * m[metric_name]

    examples = [num_examples for num_examples, _ in metrics]

    for metric_name in results.keys():
        results[metric_name] /= sum(examples)

    return results


def get_on_fit_config():
    def on_fit_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        if config.CLIENT_TYPE == config.ClientTypes.FED_PROX:
            fit_config["drop_client"] = False

        return fit_config

    return on_fit_config_fn


def get_on_eval_config():
    def on_eval_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        return fit_config

    return on_eval_config_fn
