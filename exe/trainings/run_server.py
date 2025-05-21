import socket
import sys
import torch
import logging

from src.fl.strategies import strategy_from_string
from src.deep_learning.models import UNet
import flwr as fl
from configs import config

if __name__ == "__main__":
    if config.LOCAL:
        server_address = f"0.0.0.0:8088"
        strategy_name = "fedmri"
    else:
        if len(sys.argv) > 1:
            port_number = sys.argv[1]
        else:
            port_number = config.PORT

        server_address = f"{socket.gethostname()}:{port_number}"

        if len(sys.argv) > 2:
            strategy_name = sys.argv[2]
        else:
            strategy_name = "fedavg"
            logging.info("Strategy not provided. FedAvg taken as the default one")

    logging.info("Server start")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet = UNet().to(device)

    strategy = strategy_from_string(unet, strategy_name)

    logging.info("\n\nSERVER STARTING...")
    logging.info("Strategy utilized: {}".format(strategy))
    logging.info("Server address: {}\n".format(server_address))

    if not config.LOCAL:
        node_filename = f"{config.NODE_SERVER_DIRPATH}/{strategy_name}{config.NODE_FILENAME}"
        logging.info(f"Node id saved to {node_filename}")
        with open(node_filename, 'w') as file:
            file.write(socket.gethostname())

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config.N_ROUNDS),
        strategy=strategy
    )
