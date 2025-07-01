# FL-Varying-Normalization

Deep learning project for federated learning with varying normalization techniques.

## Installation
The project requires Python 3.8+ and PyTorch. All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Configuration
The main configuration file is `configs/config.py`. Key settings include:

```python
# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_EPOCHS_CENTRALIZED = 10  # epochs for classical training

# FL Parameters
N_ROUNDS = 32
MIN_FIT_CLIENTS = MIN_AVAILABLE_CLIENTS = 2
FRACTION_FIT = 1.0

# Data Settings
USED_MODALITIES = ["t1", "t2", "flair"]  # MRI modalities
MASK_DIR = 'mask'  # Directory containing mask files
```

## Running Training

### Classical (Centralized) Training
Train on a single dataset:
```bash
python exe/trainings/classical_train.py <data_directory> [num_epochs]
```

Train on all available datasets:
```bash
python exe/trainings/classical_train.py all [num_epochs]
```

### Federated Learning

1. Start the server:
```bash
python exe/trainings/run_server.py [port_number] [strategy_name]
# Default: port=8088, strategy=fedavg
```

2. Start clients (run multiple times for different clients):
```bash
python exe/trainings/run_client_train.py <data_directory> <client_id> <server_address> <strategy_name>
```

Example local setup:
```bash
# Start server
python exe/trainings/run_server.py 8088 fedavg

# Start client
python exe/trainings/run_client_train.py /path/to/data 1 127.0.0.1:8088 fedavg
```