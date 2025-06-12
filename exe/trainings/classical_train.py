import sys
from shutil import copy2

from src.deep_learning.datasets import *
from src.deep_learning.models import *
from src.utils.files_operations import get_youngest_dir

from torch.utils.data import DataLoader

if __name__ == '__main__':
    # setting default parameters
    pretrained_model_path = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define the paths and number of epochs
    if config.LOCAL:
        train_directories = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\small_only_mask"
        validation_directory = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test\\small_no_mask"
        pretrained_model_path = "C:\\Users\\JanFiszer\\repos\\fl-varying-normalization\\trained_models\\st\\model-zscore-MSE_DSSIM-ep2-lr0.001-GN-2025-04-24-8h\\best_model.pth"
        num_epochs = config.N_EPOCHS_CENTRALIZED

    else:
        data_dir = sys.argv[1]
        if data_dir == "all":
            data_root_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/varying-normalization/data"
            train_directories = [os.path.join(data_root_dir, data_inner_dir, config.TRAIN_DIR_NAME) for data_inner_dir in os.listdir(data_root_dir)]
            validation_directory =[os.path.join(data_root_dir, data_inner_dir, config.VAL_DIR_NAME) for data_inner_dir in os.listdir(data_root_dir)]
        else:
            train_directories = os.path.join(data_dir, config.TRAIN_DIR_NAME)
            validation_directory = os.path.join(data_dir, config.VAL_DIR_NAME)

        if len(sys.argv) > 2:
            num_epochs = int(sys.argv[2])
        else:
            num_epochs = config.N_EPOCHS_CENTRALIZED

    # creating datasets
    train_dataset = SegmentationDataset2DSlices(train_directories, config.USED_MODALITIES, config.MASK_DIR, binarize_mask=True)
    validation_dataset = SegmentationDataset2DSlices(validation_directory, config.USED_MODALITIES, config.MASK_DIR, binarize_mask=True)

    # setting dataloaders
    if config.LOCAL:
        trainloader = DataLoader(train_dataset,
                                 batch_size=4,
                                 shuffle=True)
        valloader = DataLoader(validation_dataset,
                               batch_size=1,
                               shuffle=True)
    else:
        if config.PLOT_BATCH_WITH_METRICS:
            val_batch_size = 1
        else:
            val_batch_size = config.BATCH_SIZE

        num_workers = config.NUM_WORKERS
        logging.info(f"Training with {num_workers} num_workers.")
        logging.info(f"Batch size for validation set {val_batch_size}.")

        trainloader = DataLoader(train_dataset,
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=config.NUM_WORKERS,
                                 pin_memory=True)
        valloader = DataLoader(validation_dataset,
                               batch_size=val_batch_size,
                               shuffle=True,
                               num_workers=config.NUM_WORKERS,
                               pin_memory=True)

    # extract the `representative_dir` based on the data it is trained
    if isinstance(train_directories, list):
        representative_dir = "all"
    else:
        representative_dir = get_youngest_dir(train_directories)

    # create the model_dir name name having some config info and mkdir it 
    model_dir = f"{config.DATA_ROOT_DIR}/trained_models/model-{representative_dir}-ep{num_epochs}-lr{config.LEARNING_RATE}-{config.NORMALIZATION.name}-{config.now.date()}-{config.now.hour}h"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # initialize the UNet model and the criterion
    criterion = metrics.LossGeneralizedTwoClassDice(device)
    unet = UNet(criterion).to(device)

    # get the pretrained weights (if provided)
    if pretrained_model_path:
        unet.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
        model_dir = os.path.dirname(pretrained_model_path)

    # initialize the optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=config.LEARNING_RATE)

    # copy the config to the model_dir for further investigation
    config_path = "./configs/config.py"
    try:
        copy2(config_path, f"{model_dir}/config.py")
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}. You are in {os.getcwd()}")

    # train the model
    if config.LOCAL:
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=config.N_EPOCHS_CENTRALIZED,
                           plots_dir="new_metrics_local_visualization",
                           model_dir=model_dir,
                           save_best_model=False
                           # filename="model.pth",
                           )
    else:
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=num_epochs,
                           model_save_filename="last_epoch_model.pth",
                           model_dir=model_dir,
                           history_filename="history.pkl",
                           plots_dir="visualization",
                           save_best_model=True)
