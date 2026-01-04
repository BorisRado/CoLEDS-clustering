import os
import json
import hashlib
import fcntl
from functools import wraps

import pandas as pd
import wandb
from hydra.core.config_store import OmegaConf


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
    config = OmegaConf.to_container(cfg)

    profiler_name = cfg.profiler

    config["profiler_name"] = profiler_name
    log_keys = cfg.wandb.loggin_keys
    lv = [f"{k}{get_value(config, k)}" for k in log_keys]

    exp_name = "_".join([profiler_name] + lv)
    flat_config = flatten_dict(config)

    if cfg.wandb.log_to_wandb:
        wandb.init(
            config=flat_config,
            name=exp_name
        )


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


def lock_folder(folder_path):
    """Decorator that locks a folder before function execution and releases it after."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            lock_file_path = os.path.join(folder_path, ".lock")

            # Use file locking to ensure only one process accesses the folder at a time
            with open(lock_file_path, "w") as lock_file:
                try:
                    # Acquire exclusive lock
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

                    # Call the original function with folder parameter
                    return func(*args, folder=folder_path, **kwargs)

                finally:
                    # Lock is automatically released when file is closed
                    pass
        return wrapper
    return decorator


@lock_folder(".tmp_cache")
def run_exists_already(config, folder):
    new_conf = config
    if not isinstance(config, dict):
        new_conf = flatten_dict(OmegaConf.to_container(config))

    if not new_conf["wandb.log_to_wandb"]:
        return False

    del new_conf["experiment.folder"]
    if "profiler_name" in new_conf:
        del new_conf["profiler_name"]

    # Check if configuration already exists
    for filename in os.listdir(folder):
        if filename == ".lock":  # Skip lock file
            continue
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r") as fp:
                existing_config = json.load(fp)
            if existing_config == new_conf:
                return True

    # save the dictionary with a unique filename based on config hash
    config_str = json.dumps(new_conf, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    filename = os.path.join(folder, f"config_{config_hash}.json")

    # Double-check that file doesn't exist (in case of clashes)
    assert not os.path.exists(filename), f"FILE: {str(filename)}"
    with open(filename, "w") as f:
        json.dump(new_conf, f, sort_keys=True, indent=2)
        f.write("\n")

    return False
