import os
import json
from pathlib import Path

import wandb
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv


COLEDS_CONFIG_TO_SHORT_NAME = {
    "dataset.dataset_name": "dataset",
    "model._target_": "model",
    "train_config.batch_size": "batch_size",
    "train_config.fraction_fit": "fraction_fit",
    "train_config.num_client_updates": "num_client_updates",
    "train_config.temperature": "temperature",
    "general.seed": "seed",
}

DATASET_NAME_MAPPING = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "flwrlabs/cinic10": "CINIC10",
    "mnist": "MNIST",
    "zalando-datasets/fashion_mnist": "Fashion-MNIST",
}

MODEL_NAME_MAPPING = {
    "src.models.set2set_model.Set2SetModel": "Set2Set",
    "src.models.clmean_model.ClMeanModel": "Cl-Mean",
    "src.models.gru.GRUModel": "GRU",
}


def get_run_dataframe(run, dry=False):
    project = os.environ.get("WANDB_PROJECT")
    run_folder = Path("data/raw") / project
    run_folder.mkdir(exist_ok=True, parents=True)
    if isinstance(run, str):
        run_id = run
    else:
        run_id = run.id
    run_csv = run_folder / f"{run_id}.csv"
    run_config_json = run_folder / f"{run_id}_config.json"
    if dry and run_csv.exists() and run_config_json.exists():
        return
    try:
        df = pd.read_csv(run_csv)
        with open(run_config_json, "r") as fp:
            run_config = json.load(fp)
        assert run_config == run.config
    except FileNotFoundError:
        df = run.history()
        df.to_csv(run_csv, index=False)
        with open(run_config_json, "w") as fp:
            json.dump(run.config, fp)
    return df


def get_project_runs():
    load_dotenv(override=True)
    os.environ["WANDB_SILENT"] = "true"
    project = os.environ["WANDB_PROJECT"]

    wandb.Settings(disable_job_creation=True)
    api = wandb.Api(timeout=60)
    runs = api.runs(
        path=f"borisrado/{project}",
        order="+created_at",
        per_page=1024,
        filters={"state": "finished"}
    )
    print(len(runs), "runs found")
    return runs


def download_all_data():
    runs = get_project_runs()
    for run in tqdm(runs):
        _ = get_run_dataframe(run, dry=True)


def get_coleds_dataframe(refresh=False):
    save_df_path = "data/all_coleds_best_correlations.csv"
    if not refresh:
        return pd.read_csv(save_df_path)
    runs = get_project_runs()
    data = []
    for run in runs:
        config = run.config
        if "train_config.fraction_fit" not in config:
            continue
        df = get_run_dataframe(run)
        record = {
            v: config[k] for k, v in COLEDS_CONFIG_TO_SHORT_NAME.items()
        }
        record["max_correlation"] = df["correlation"].max().item()
        data.append(record)
    df = pd.DataFrame(data)

    df["dataset"] = df["dataset"].map(DATASET_NAME_MAPPING)
    df["model"] = df["model"].map(MODEL_NAME_MAPPING)

    # sanity check
    # assert (df.groupby(
    #     ["dataset", "model", "batch_size", "fraction_fit", "num_client_updates", "temperature"]
    # ).size() == 3).all()

    df.to_csv(save_df_path, index=False)
    return df


def filter_df(df, filters):
    s = pd.Series([True] * df.shape[0])
    remove_keys = []
    for k, v in filters.items():
        if isinstance(v, set):
            s &= df[k].isin(v)
        else:
            s &= df[k] == v
            remove_keys.append(k)

    df = df[s].copy()
    return df.drop(remove_keys, axis=1)


def plot_with_std(df, index, columns, ax, index_order=None, **kwargs):
    df = df.pivot(index=index, columns=columns, values=["mean", "std"]).copy()
    min_value = df["mean"].min().min()
    max_value = df["mean"].max().max()
    if index_order is not None:
        df = df.loc[index_order]
    df["mean"].plot(kind="bar", yerr=df["std"], rot=0, ylim=(min_value-0.02, max_value+0.02), ax=ax, **kwargs)
