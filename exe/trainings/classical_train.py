import os
import sys
from shutil import copy2
from pathlib import Path
import torch

from src.deep_learning.datasets import *
from src.deep_learning.models import *
from src.utils.files_operations import get_youngest_dir

from torch.utils.data import DataLoader

if __name__ == '__main__':

    if config.LOCAL:
        train_directory = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\small"
        validation_directory = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\small"
    else:
        data_dir = sys.argv[1]
        train_directory = os.path.join(data_dir, "train")
        validation_directory = os.path.join(data_dir, "validation")
        representative_test_dir = train_directory[0].split(os.path.sep)[-2]
        if len(sys.argv) > 2:
            num_epochs = int(sys.argv[2])
        else:
            num_epochs = config.N_EPOCHS_CENTRALIZED

    train_dataset = SegmentationDataset2DSlices(train_directory, config.USED_MODALITIES, config.MASK_DIR, binarize_mask=True)
    validation_dataset = SegmentationDataset2DSlices(validation_directory, config.USED_MODALITIES, config.MASK_DIR, binarize_mask=True)

    if config.LOCAL:
        trainloader = DataLoader(train_dataset,
                                 batch_size=2)
        valloader = DataLoader(validation_dataset,
                               batch_size=1)
    else:
        num_workers = config.NUM_WORKERS
        print(f"Training with {num_workers} num_workers.")

        trainloader = DataLoader(train_dataset,
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=config.NUM_WORKERS,
                                 pin_memory=True)
        valloader = DataLoader(validation_dataset,
                               batch_size=config.BATCH_SIZE,
                               shuffle=True,
                               num_workers=config.NUM_WORKERS,
                               pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # criterion = metrics.BinaryDiceLoss()
    criterion = metrics.loss_generalized_dice

    unet = UNet(criterion).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=config.LEARNING_RATE)

    representative_test_dir = get_youngest_dir(train_directory)
    model_dir = f"{config.DATA_ROOT_DIR}/trained_models/model-{representative_test_dir}-{config.LOSS_TYPE.name}-ep{num_epochs}-lr{config.LEARNING_RATE}-{config.NORMALIZATION.name}-{config.now.date()}-{config.now.hour}h"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    logging.warning("OVERWRITING THE MODELDIR")
    
    try:
        config_path = "./configs/config.py"
        copy2(config_path, f"{model_dir}/config.py")
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}. You are in {os.getcwd()}")


    if config.LOCAL:
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=config.N_EPOCHS_CENTRALIZED,
                           plots_dir="visualization",
                           model_dir=model_dir,
                           history_filename="history.pkl"
                           # filename="model.pth",
                           )
    else:
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=num_epochs,
                           model_save_filename="last_epoch_model.pth",
                           model_dir=model_dir,
                           history_filename="history.pkl",
                           plots_dir="predictions",
                           save_best_model=True)
