import os

import wandb
from tqdm import tqdm
from dotenv import load_dotenv

from src.utils.wandb import run_exists_already


def run():
    folder = ".tmp_cache"
    api = wandb.Api(timeout=30)
    runs = api.runs(per_page=100000, filters={
        **{"state": "finished"},
    })
    os.makedirs(folder)

    for run in tqdm(runs):
        config = run.config
        assert not run_exists_already(config)


if __name__ == "__main__":
    load_dotenv()
    run()
