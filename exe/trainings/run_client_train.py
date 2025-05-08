import sys
import logging
from configs import config
from src.fl.clients import *
from src.deep_learning import metrics, models

if __name__ == "__main__":
    # moving on ares/athena to the repo directory
    if config.LOCAL:
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\segmentation_ucsf_whitestripe_test"
        client_id = "1"
        server_address = "127.0.0.1:8088"
        with_num_workers = False
    else:
        data_dir = sys.argv[1]
        client_id = sys.argv[2]
        server_address = sys.argv[3]
        strategy_name = sys.argv[4]
        with_num_workers = True

        if not config.LOCAL:
            with open(f"{config.NODE_SERVER_DIRPATH}/{strategy_name}{config.NODE_FILENAME}", 'r') as file:   # TODO: sys.argv
                server_node = file.read()

        if ":" not in server_address:
            server_address = f"{server_node}:{server_address}"

    logging.info(f"Client start {client_id}")
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = metrics.LossGeneralizedTwoClassDice(device)
    unet = models.UNet(criterion).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=config.LEARNING_RATE)
    client = client_from_string(client_id, unet, optimizer, data_dir, sys.argv[4])

    logging.info(f"The retrieved server address is : {server_address}")

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
