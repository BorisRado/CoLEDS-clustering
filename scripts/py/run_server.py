import logging
logging.basicConfig(level=logging.ERROR)

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
from flwr.server import ServerConfig
from slwr.server.app import start_server

from src.utils.stochasticity import TempRng, set_seed
from src.slower.strategy import Strategy
from src.slower.server_model import ServerModel


@hydra.main(version_base=None, config_path="../../conf", config_name="cl")
def run(cfg):
    server_model_fn = lambda: ServerModel(cfg.train_config.temperature)
    num_clients = cfg.partitioning.num_partitions

    with TempRng(cfg.general.seed):
        model = instantiate(cfg.model, input_shape=cfg.dataset.input_shape)

    num_clients = cfg.num_clients
    strategy = Strategy(
        model=model,
        optim_kwargs=OmegaConf.to_container(cfg.optimizer),
        evaluation_freq=cfg.train_config.num_iterations,
        fraction_fit=cfg.train_config.fraction_fit,
        fraction_evaluate=0.,
        init_server_model_fn=server_model_fn,
        min_available_clients=num_clients,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        common_server_model=True,
        process_clients_as_batch=True,
    )

    set_seed(cfg.general.seed)
    history = start_server(
        server_address="0.0.0.0:50051",
        strategy=strategy,
        config=ServerConfig(num_rounds=cfg.num_rounds),
    )
    print(history)
    print(strategy.losses)


if __name__ == "__main__":
    run()
