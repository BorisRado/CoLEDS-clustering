import torch
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.utils.stochasticity import TempRng
from src.cem.single_model_cem import SingleModelCEM
from src.cem.es_cem import EmbeddingSpaceCEM


def load_cem(cem_config, cem_folder):
    n_classes = cem_config.dataset.n_classes

    if "cem" in cem_config:
        # weight difference
        cem_name = cem_config["cem"]["_target_"].split(".")[-1]
        print(f"Loading {cem_name}")
        if cem_name == "WeightDiffCEM":
            with TempRng(cem_config.general.seed):
                model = instantiate(
                    cem_config.model,
                    input_shape=cem_config.dataset.input_shape,
                    n_classes=n_classes
                )
            cem = instantiate(
                cem_config.cem,
                init_model=model,
                optim_kwargs=OmegaConf.to_container(cem_config.optimizer),
            )
        elif cem_name == "LogitCEM":
            with TempRng(cem_config.general.seed):
                model = instantiate(
                    cem_config.model,
                    input_shape=cem_config.dataset.input_shape,
                    n_classes=n_classes
                )
            cem = instantiate(
                cem_config.cem,
                init_model=model,
                optim_kwargs=OmegaConf.to_container(cem_config.optimizer),
            )

        # embedding space
        elif cem_name == "EmbeddingSpaceCEM":
            model = instantiate(
                cem_config["model"],
                input_shape=cem_config.dataset.input_shape,
                n_classes=n_classes
            )
            model.load_state_dict(torch.load(cem_folder / "fl_model.pth"))
            cem = EmbeddingSpaceCEM(model, cem_config["cem"]["reduction_stats"])
        # random & label
        elif cem_name in {"RandomCEM", "LabelCEM"}:
            cem = instantiate(cem_config["cem"], n_classes=n_classes)
        else:
            raise Exception(str(cem_config))
    else:
        # is contrastive learning model
        model = instantiate(cem_config.model, input_shape=cem_config.dataset.input_shape)
        model.load_state_dict(torch.load(cem_folder / "cem.pth"))
        cem = SingleModelCEM(model)
    return cem
