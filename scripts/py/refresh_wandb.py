from tqdm import tqdm

import wandb
import os
import json


def run():
    api = wandb.Api(timeout=30)
    runs = api.runs(per_page=100000, filters={
        **{"state": "finished"},
    })
    configs = [(run.id, run.config) for run in runs]
    os.makedirs(".wandb_cache", exist_ok=True)

    for run_id, config in tqdm(configs):
        with open(f".wandb_cache/{run_id}.json", "w") as f:
            json.dump(config, f)


if __name__ == "__main__":
    run()
