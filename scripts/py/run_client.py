import os
import logging
logging.basicConfig(level=logging.ERROR)

import hydra
from hydra.utils import instantiate
from slwr.client.app import start_client

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
    else:
        assert False  # TODO: colext
        client_idx = os.getenv("CLIENT_IDX", 0)

    trainsets, valsets = get_datasets_from_cfg(cfg)
    trainset, valset = trainsets[client_idx], valsets[client_idx]

    trainset = to_pytorch_tensor_dataset([trainset])[0]
    valset = to_pytorch_tensor_dataset([valset])[0]

    model = instantiate(cfg.model, input_shape=cfg.dataset.input_shape)

    client = Client(
        model,
        trainset,
        valset,
        batch_size=cfg.train_config.batch_size,
        num_client_updates=cfg.train_config.num_client_updates
    )

    set_seed(cfg.general.seed)
    start_client(server_address=f"{cfg.server_ip}:50051", client=client)

if __name__ == "__main__":
    run()
