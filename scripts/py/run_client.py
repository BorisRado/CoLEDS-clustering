import os
import logging
logging.basicConfig(level=logging.ERROR)

import hydra
from hydra.utils import instantiate
from slwr.client.app import start_client

from src.utils.colext import EnvironmentVariables as EV
from src.data.utils import (
    get_datasets_from_cfg,
    to_pytorch_tensor_dataset
)
from src.utils.stochasticity import set_seed
from src.slower.client import Client

@hydra.main(version_base=None, config_path="../../conf", config_name="cl")
def run(cfg):
    if "client_idx" in cfg:
        client_idx = cfg.client_idx
        server_ip = cfg.server_ip
    else:
        server_ip = os.environ[EV.SERVER_ADDRESS]
        client_idx = int(os.environ[EV.CLIENT_ID])

    trainsets, valsets = get_datasets_from_cfg(cfg)
    trainset, valset = trainsets[client_idx], valsets[client_idx]

    trainset = to_pytorch_tensor_dataset([trainset], to_cuda=False)[0]
    valset = to_pytorch_tensor_dataset([valset], to_cuda=False)[0]

    model = instantiate(cfg.model, input_shape=cfg.dataset.input_shape)

    client = Client(
        model,
        trainset,
        valset,
        batch_size=cfg.train_config.batch_size,
        num_client_updates=cfg.train_config.num_client_updates
    )

    set_seed(cfg.general.seed)
    start_client(server_address=server_ip, client=client)

if __name__ == "__main__":
    run()
