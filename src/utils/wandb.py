import json
from pathlib import Path

import pandas as pd
import wandb
from hydra.core.config_store import OmegaConf
from dotenv import load_dotenv


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_value(config, key):
    key_list = key.split(".")
    v = config
    for k in key_list:
        v = v[k]
    return v


def init_wandb(cfg):
    load_dotenv(override=True)
    config = OmegaConf.to_container(cfg)

    try:
        cem_name = cfg.cem._target_.split(".")[-1]
    except:
        cem_name = cfg.model._target_.split(".")[-1]

    config["cem_name"] = cem_name
    log_keys = cfg.wandb.loggin_keys
    lv = [f"{k}{get_value(config, k)}" for k in log_keys]

    exp_name = "_".join([cem_name] + lv)
    flat_config = flatten_dict(config)

    if cfg.wandb.log_to_wandb:
        run = wandb.init(
            config=flat_config,
            name=exp_name
        )
        run_id = run.id
    else:
        run_id = str(cfg.temp_run_id)
    exp_path = Path("data/raw") / run_id
    exp_path.mkdir(parents=True, exist_ok=not cfg.wandb.log_to_wandb)
    OmegaConf.save(cfg, exp_path / "config.yaml")

    return exp_path


def log_table(d, experiment_folder, table_name, **kwargs):
    df = pd.DataFrame.from_dict(d)
    df.to_csv(experiment_folder / f"{table_name}.csv", index=False)
    if wandb.run is not None:
        wandb.log({
            "table": wandb.Table(dataframe=df),
        } | kwargs)


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()
