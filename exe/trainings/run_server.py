import socket
import sys
import torch

from src.fl.strategies import strategy_from_string
from src.deep_learning.models import UNet
import flwr as fl
from configs import config

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet = UNet().to(device)

    strategy = strategy_from_string(unet, "fedavg")

    if config.LOCAL:
        server_address = f"0.0.0.0:8088"
    else:
        if len(sys.argv) > 1:
            port_number = sys.argv[1]
        else:
            port_number = config.PORT

        server_address = f"{socket.gethostname()}:{port_number}"

    print("\n\nSERVER STARTING...")
    print("Strategy utilized: {}".format(strategy))
    print("Server address: {}\n".format(server_address))

    if not config.LOCAL:
        with open(f"server_nodes/{sys.argv[2]}{config.NODE_FILENAME}", 'w') as file:
            file.write(socket.gethostname())

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config.N_ROUNDS),
        strategy=strategy
    )
