import logging
from os import path
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex

from configs import config, creds
from src.deep_learning import metrics
from src.utils import files_operations as fop, visualization

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_print_freq = config.BATCH_PRINT_FREQ

mse = nn.MSELoss()

dice_generalized = metrics.generalized_dice
dice_2_class = metrics.dice_2_class
# dice_score = Dice().to(device)
generalized_dice_torchmetrics = metrics.GeneralizedDiceScore(2).to(device)
# dice_score = Dice().to(device)
jaccard_index = BinaryJaccardIndex().to(device)


class UNet(nn.Module):
    """
        UNet model class used for federated learning.
        Consists the methods to train and evaluate.
        Allows two different normalizations: 
            - standard BatchNormalization
            - GroupNorm (the number of groups specified in the config)
    """

    def __init__(self, criterion=None, descriptive_metric = None, bilinear=False, normalization=config.NORMALIZATION):
        super(UNet, self).__init__()

        self.criterion = criterion
        self.bilinear = bilinear
        if descriptive_metric:
            self.descriptive_metric = descriptive_metric
        else:
            self.descriptive_metric = "loss"


        self.available_metrics = {"loss": self.criterion,
                                  "mse": mse,
                                  "generalized_dice_torchmetrics": generalized_dice_torchmetrics,
                                  # "dice_classification": dice_score,
                                  "dice_generalized": dice_generalized,
                                  "dice_2_class": dice_2_class,
                                  "jaccard": jaccard_index}

        self.inc = (DoubleConv(3, 64, normalization))
        self.down1 = (Down(64, 128, normalization))
        self.down2 = (Down(128, 256, normalization))
        self.down3 = (Down(256, 512, normalization))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, normalization))
        self.up1 = (Up(1024, 512 // factor, normalization, bilinear))
        self.up2 = (Up(512, 256 // factor, normalization, bilinear))
        self.up3 = (Up(256, 128 // factor, normalization, bilinear))
        self.up4 = (Up(128, 64, normalization, bilinear))
        self.outc = (OutConv(64, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def save(self, dir_name: str, filename=None):
        """
        Saves the model to a given directory. Allows to change the name of the file, by default it is "model".
        """
        if filename is None:
            filename = "model"

        if not isinstance(dir_name, str):
            raise TypeError(f"Given directory name {dir_name} has wrong type: {type(dir_name)}.")

        Path(dir_name).mkdir(exist_ok=True)

        if not filename.endswith(".pth"):
            filename += ".pth"

        filepath = f"{dir_name}/{filename}"
        torch.save(self.state_dict(), filepath)

        logging.info(f"Model saved to: {fi}")

    def _train_one_epoch(self, trainloader, optimizer):
        """
        Method used by perform_train(). Does one iteration of training.
        """
        utilized_metrics = {metric_name: self.available_metrics[metric_name] for metric_name in config.METRICS}

        epoch_metrics = {metric_name: 0.0 for metric_name in utilized_metrics.keys()}
        total_metrics = {metric_name: 0.0 for metric_name in utilized_metrics.keys()}

        n_batches = len(trainloader)

        start = time.time()
        n_train_steps = 0

        if n_batches < config.BATCH_PRINT_FREQ:
            batch_print_frequency = n_batches
        else:
            batch_print_frequency = config.BATCH_PRINT_FREQ

        for index, data in enumerate(trainloader):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = self(images)
            loss = self.criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            for metric_name, metric_object in utilized_metrics.items():
                if metric_name == "loss":
                    metric_value = loss
                else:
                    metric_value = metric_object(predictions, targets.int())

                total_metrics[metric_name] += metric_value.item()
                epoch_metrics[metric_name] += metric_value.item()

            n_train_steps += 1

            divided_batch_metrics = {metric_name: total_value / batch_print_frequency for metric_name, total_value
                                     in total_metrics.items()}
            metrics_str = metrics.metrics_to_str(divided_batch_metrics, starting_symbol="\t")

            if index % batch_print_frequency == batch_print_frequency - 1:
                logging.info(f'\t\tbatch {(index + 1)} out of {n_batches}\t\t{metrics_str}')
            else:
                logging.debug(f'\t\tbatch {(index + 1)} out of {n_batches}\t\t{metrics_str}')

            total_metrics = {metric_name: 0.0 for metric_name in utilized_metrics.keys()}

        averaged_epoch_metrics = {metric_name: metric_value / n_train_steps for metric_name, metric_value in
                                  epoch_metrics.items()}
        metrics_epoch_str = metrics.metrics_to_str(averaged_epoch_metrics, starting_symbol="")

        logging.info(f"\tTime exceeded: {time.time() - start:.1f}")
        logging.info(f"\tEpoch metrics: {metrics_epoch_str}")

        return averaged_epoch_metrics

    def perform_train(self,
                      trainloader,
                      optimizer,
                      epochs,
                      validationloader=None,
                      model_dir=None,
                      model_save_filename=None,
                      history_filename=None,
                      plots_dir=None,
                      save_best_model=False,
                      save_each_epoch=False):
        """
            Performs the train for a given number of epochs.
        """
        logging.info(f"\n\tTRAINING... \n\ton device: {device} \n\twith loss: {self.criterion}\n")
        # TODO: automatic model_dir
        if model_dir is None:
            model_dir = f"{config.DATA_ROOT_DIR}/trained_models/gen-model-{config.LOSS_TYPE.name}-ep{epochs}-lr{config.LEARNING_RATE}-{config.NORMALIZATION.name}-{config.now.date()}-{config.now.hour}h"

        if config.USE_WANDB:
            wandb.login(key=creds.api_key_wandb)
            wandb.init(
                name=model_dir.split(path.sep)[-1],  # keeping only the last part of the model_dir (it stores all the viable information)
                project=config.PROJECT_NAME)  # TODO: config as dict

        if not isinstance(self.criterion, Callable):
            raise TypeError(f"Loss function (criterion) has to be callable. It is {type(self.criterion)} which is not.")

        if any([history_filename, plots_dir, model_save_filename]):
            logging.info(f"\tModel, history and plots will be saved to {model_dir}")
        else:
            logging.warning(f"\tNeither model, history nor plots from the training process will be saved!")

        val_metric_names = [f"val_{m_name}" for m_name in config.METRICS]

        history = {m_name: [] for m_name in config.METRICS}
        history.update({m_name: [] for m_name in val_metric_names})

        if save_best_model:
            best_loss = float('inf')

        if plots_dir is not None:
            plots_path = path.join(model_dir, plots_dir)
            Path(plots_path).mkdir(parents=True, exist_ok=True)

        else:
            plots_path = None

        for epoch in range(epochs):
            logging.info(f"\tEPOCH: {epoch + 1}/{epochs}")

            epoch_metrics = self._train_one_epoch(trainloader, optimizer)

            for metric in config.METRICS:
                history[metric].append(epoch_metrics[metric])

            logging.info("\tVALIDATION...")
            if validationloader is not None:
                val_metric = self.evaluate(validationloader, plots_path, plot_every_batch_with_metrics=config.PLOT_BATCH_WITH_METRICS, epoch_number=epoch)

                for metric in val_metric_names:
                    # trimming after val_ to get only the metric name since it is provided by the
                    history[metric].append(val_metric[metric])

                if save_best_model:
                    if val_metric["val_loss"] < best_loss:
                        logging.debug(
                            f"\tModel form epoch {epoch} taken as the best one.\n\tIts loss {val_metric['val_loss']:.3f} is better than current best loss {best_loss:.3f}.")
                        best_loss = val_metric["val_loss"]
                        best_model = self.state_dict()

                if config.USE_WANDB:
                    wandb.log(val_metric)

            if config.USE_WANDB:
                wandb.log(epoch_metrics)

            if save_each_epoch:
                self.save(model_dir, f"model-ep{epoch}.pth")

            if history_filename is not None:
                with open(path.join(model_dir, history_filename), 'wb') as file:
                    pickle.dump(history, file)

        # saving best model
        if save_best_model:
            torch.save(best_model, path.join(model_dir, "best_model.pth"))

        logging.info("\tAll epochs finished.\n")

        if model_save_filename is not None:
            self.save(model_dir, model_save_filename)

        if config.USE_WANDB:
            wandb.finish()

        return history

    def evaluate(self,
                 testloader,
                 plots_path=None,
                 compute_std=False,
                 wanted_metrics=None,
                 save_preds_dir=None,
                 plot_metrics_distribution=False,
                 plot_every_batch_with_metrics=False,
                 epoch_number=None
                 # high_mse_value=float('inf')
                 ):
        if not isinstance(self.criterion, Callable):
            raise TypeError(f"Loss function (criterion) has to be callable. It is {type(self.criterion)} which is not.")
        if compute_std and testloader.batch_size != 1:
            raise ValueError("The computations will result in wrong results! Batch size should be 1 if `compute_std=True`.")
        if epoch_number is None:
            epoch_number = "last_epoch"
        else:
            epoch_number = f"ep_{epoch_number}"

        logging.info(f"\t\tON DEVICE: {device} \n\t\t\t\tWITH LOSS: {self.criterion}\n")

        n_steps = 0
        n_skipped = 0
        utilized_metrics = {metric_name: self.available_metrics[metric_name] for metric_name in config.METRICS}

        if wanted_metrics:
            utilized_metrics = {metric_name: metric_obj for metric_name, metric_obj in utilized_metrics.items() if
                       metric_name in wanted_metrics}

        utilized_metrics = {f"val_{name}": metric for name, metric in utilized_metrics.items()}

        if save_preds_dir:
            Path(save_preds_dir).mkdir(exist_ok=True)

        metrics_values = {m_name: [] for m_name in utilized_metrics.keys()}
        with torch.no_grad():
            for batch_index, batch in enumerate(testloader):
                # loading the input and target images
                images_cpu, targets_cpu = batch[0], batch[1]

                images = images_cpu.to(device)
                targets = targets_cpu.to(device)

                # utilizing the network
                predictions = self(images)

                # saving all the predictions
                if save_preds_dir:
                    current_batch_size = images.shape[0]

                    if batch_index == 0:  # getting the batch size in the first round
                        batch_size = current_batch_size

                    for img_index in range(current_batch_size):  # iterating over current batch size (number of images)
                        # retrieving the name of the current slice
                        patient_slice_name = \
                            testloader.dataset.images[img_index + batch_index * batch_size].split(path.sep)[-1]
                        pred_filepath = path.join(save_preds_dir, patient_slice_name)

                        # saving the current image to the declared directory with the same name as the input image name
                        logging.debug(f"\t\t\tPrediction (batch={batch_index}, slice_index={img_index}) will be saved to: {pred_filepath}")
                        np.save(pred_filepath, predictions[img_index].cpu().numpy())

                # calculating metrics
                for metric_name, metric_obj in utilized_metrics.items():

                    metric_value = metric_obj(predictions, targets)
                    metrics_values[metric_name].append(metric_value.item())

                if plot_every_batch_with_metrics:
                    if testloader.batch_size == 1:
                        if plots_path:
                            # if isinstance(metric_obj, metrics.GeneralizedDiceLoss) and metric_value.item() > high_mse_value:

                            batches_with_metrics_dirpath = path.join(plots_path, f"batches_with_metrics_ep{epoch_number}")
                            Path(batches_with_metrics_dirpath).mkdir(exist_ok=True)

                            descriptive_metric_value = metrics_values[self.descriptive_metric][-1]
                            filepath = path.join(batches_with_metrics_dirpath, f"slice{batch_index}_{self.descriptive_metric}{descriptive_metric_value:.2f}.jpg")
                            current_batch_metrics = {metric_name: metrics_values[metric_name][-1] for metric_name in metrics_values.keys()}

                            visualization.plot_all_modalities_and_target(
                                images.to('cpu'), targets.to('cpu'), predictions.to('cpu').detach(),
                                title=metrics.metrics_to_str(current_batch_metrics, sep=";"),
                                savepath=filepath
                                )
                        else:
                            raise ValueError("To have plot every batch with metrics (`plot_every_batch_with_metrics=True`) the `plots_path` need to be provided")

                    else:
                        raise logging.error("To have plot every batch with metrics (`plot_every_batch_with_metrics=True`) the in DataLoader `batch_size=1`")
                n_steps += 1

        if plot_metrics_distribution:
            histograms_dir_path = path.join(plots_path, "histograms")
            Path(histograms_dir_path).mkdir(exist_ok=True)
            self.plot_distribution(metrics_values, histograms_dir_path)
            logging.debug("\t\t\tAll distribution histograms saved.")

        averaged_metrics, std_metrics = self._compute_average_std_metric(metrics_values, n_steps, n_skipped)
        metrics_str = metrics.metrics_to_str(averaged_metrics, sep='\t')

        logging.info(f"\t\tFor evaluation set: {metrics_str}\n")

        if compute_std:
            return averaged_metrics, std_metrics
        else:
            return averaged_metrics

    @staticmethod
    def plot_distribution(metrics_values, histograms_dir_path):
        # Plot and save histograms
        for key, values in metrics_values.items():
            plt.figure()  # Create a new figure
            plt.hist(values, bins=100, color='blue', alpha=0.7)
            plt.title(f"Histogram of {key}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            # Save to file
            output_path = path.join(histograms_dir_path, f"{key}_histogram.png")
            plt.savefig(output_path)
            plt.close()  # Close the figure to free up memory
            logging.debug(f"\t\t\t\tSaved histogram for {key} to {output_path}")
    @staticmethod
    def _compute_average_std_metric(metrics_values, n_steps, n_skipped):
        """
        Computes the average for each of the metrics using the sum and the number of steps. 
        Treats specially the ZoomedSSIM metrics since there are potential skips. 
        """
        averaged_metrics = {}
        std_metrics = {}
        for metric_name, metric_values in metrics_values.items():
            numpy_metrics_values = np.array(metric_values)
            if "zoomed_ssim" in metric_name:
                if n_skipped == n_steps:
                    logging.log(logging.WARNING, f"All the mask in the provided dataset are zeros."
                                                 "\nNone ZoomedSSIM values were computed. Result assigned to None")

                    averaged_metrics[metric_name] = None
                denominator = n_steps - n_skipped
            else:
                denominator = n_steps

            averaged_metrics[metric_name] = numpy_metrics_values.sum() / denominator
            std_metrics[metric_name] = numpy_metrics_values.std()

        return averaged_metrics, std_metrics


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        # choosing between one of three possible normalization types
        if normalization == config.LayerNormalizationType.BN:
            self.norm1 = nn.BatchNorm2d(mid_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif normalization == config.LayerNormalizationType.GN:
            self.norm1 = nn.GroupNorm(config.N_GROUP_NORM, mid_channels)
            self.norm2 = nn.GroupNorm(config.N_GROUP_NORM, out_channels)
        else:
            self.norm1 = None
            self.norm2 = None

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm1:
            x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.norm2:
            x = self.norm2(x)
        x = self.relu(x)

        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normalization):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, normalization)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, normalization, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, normalization, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, normalization)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))
