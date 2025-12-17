import os
import json
from pathlib import Path

import wandb
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


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


def get_coleds_dataframe(refresh=False, number_of_seeds=None):
    save_df_path = "data/all_coleds_best_correlations.csv"
    if not refresh:
        try:
            return pd.read_csv(save_df_path)
        except FileNotFoundError:
            print("Could not load data. Re-downloading")
            return get_coleds_dataframe(refresh=True, number_of_seeds=number_of_seeds)

    runs = get_project_runs()
    data = []
    for run in runs:
        config = run.config
        if config["profiler"] != "coleds":
            continue
        df = get_run_dataframe(run)
        record = {
            v: config[k] for k, v in COLEDS_CONFIG_TO_SHORT_NAME.items()
        }
        record["max_correlation"] = df["correlation"].max()
        data.append(record)
    df = pd.DataFrame(data)

    df["dataset"] = df["dataset"].map(DATASET_NAME_MAPPING)
    df["model"] = df["model"].map(MODEL_NAME_MAPPING)

    # sanity check
    if number_of_seeds is not None:
        assert (df.groupby(
            ["dataset", "model", "batch_size", "fraction_fit", "num_client_updates", "temperature"]
        ).size() == number_of_seeds).all()

    df.to_csv(save_df_path, index=False)
    return df


def get_es_dataframe(refresh=False, number_of_seeds=None):
    save_df_path = "data/all_es_best_correlations.csv"
    if not refresh:
        try:
            return pd.read_csv(save_df_path)
        except FileNotFoundError:
            print("Could not load data. Re-downloading")
            return get_es_dataframe(refresh=True, number_of_seeds=number_of_seeds)

    runs = get_project_runs()
    data = []
    for run in runs:
        config = run.config
        if config["profiler"] != "es":
            continue
        df = get_run_dataframe(run)
        model_name = "AutoEncoder" \
            if config["model._target_"].endswith("VariationalAutoencoder") \
                else "Classifier"
        dataset = config["dataset.dataset_name"]
        record = {
            "model": model_name,
            "dataset": dataset,
            "max_correlation": df["correlation"].max()
        }
        data.append(record)
    df = pd.DataFrame(data)
    df["dataset"] = df["dataset"].map(DATASET_NAME_MAPPING)

    # sanity check
    if number_of_seeds is not None:
        assert (df.groupby(["dataset", "model"]).size() == number_of_seeds).all()

    df.to_csv(save_df_path, index=False)
    return df


def get_wd_dataframe(refresh=False, number_of_seeds=None):
    save_df_path = "data/all_wd_best_correlations.csv"
    if not refresh:
        try:
            return pd.read_csv(save_df_path)
        except FileNotFoundError:
            print("Could not load data. Re-downloading")
            return get_wd_dataframe(refresh=True, number_of_seeds=number_of_seeds)

    runs = get_project_runs()
    data = []
    for run in runs:
        config = run.config
        if config["profiler"] != "wd":
            continue
        df = get_run_dataframe(run)
        dataset = config["dataset.dataset_name"]
        record = {
            "dataset": dataset,
            "max_correlation": df["correlation"].max()
        }
        data.append(record)
    df = pd.DataFrame(data)
    df["dataset"] = df["dataset"].map(DATASET_NAME_MAPPING)

    # sanity check
    if number_of_seeds is not None:
        assert (df.groupby("dataset").size() == number_of_seeds).all()

    df.to_csv(save_df_path, index=False)
    return df


def get_wd_correlations_late_in_training():
    runs = get_project_runs()
    wd_runs = [r for r in runs if r.config["profiler"] == "wd"]

    data = []
    for r in wd_runs:
        df = get_run_dataframe(r)
        df = df[df["table"].isna()].reset_index(drop=True)
        correlation = df[~df["correlation"].isna()].reset_index(drop=True).iloc[-1]["correlation"]
        data.append({
            "dataset": r.config["dataset.dataset_name"],
            "max_correlation": correlation
        })
    df = pd.DataFrame(data)
    df["dataset"] = df["dataset"].map(DATASET_NAME_MAPPING)
    wddf_late_training = df.groupby("dataset")["max_correlation"].agg(["mean", "std"])
    return wddf_late_training


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


def plot_with_std(df, index, columns, ax, index_order=None, column_order=None, **kwargs):
    df = df.pivot(index=index, columns=columns, values=["mean", "std"]).copy()
    min_value = df["mean"].min().min()
    max_value = df["mean"].max().max()
    if index_order is not None:
        df = df.loc[index_order]
    if column_order is not None:
        # reorder columns by the second level (the metric names) to match column_order
        # this reorders both "mean" and "std" blocks accordingly
        df = df.reindex(columns=column_order, level=1)
    df["mean"].plot(kind="bar", yerr=df["std"], rot=0, ylim=(min_value-0.02, max_value+0.02), ax=ax, **kwargs)


def set_matplotlib_configuration(fontsize=18):

    # matplotlib configuration
    plt.style.use("seaborn-v0_8-whitegrid")

    cmap = sns.color_palette("colorblind", 4)
    mpl.rcParams.update({
        "figure.dpi": 400, # so it's bigger when calling plt.show()

        # LaTeX text rendering
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "text.latex.preamble": r"""
            \usepackage{lmodern}
            \usepackage[T1]{fontenc}
        """,

        # Base font size
        "font.size": fontsize,

        # Axes
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,

        # Ticks
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,

        # Legends
        "legend.fontsize": fontsize,
        "legend.title_fontsize": fontsize,

        # Figure-level text
        "figure.titlesize": fontsize,

        # Super (figure-level) axis labels
        "figure.labelsize": fontsize,
    })
    savefig_kwargs = {
        "bbox_inches": "tight",
        "format": "pdf",
        "pad_inches": 0.,
    }
    return {
        "error_kw": {"capthick": 0.8, "elinewidth": 0.8, "capsize": 2},
        "color": [cmap[i] for i in range(4)]
    }, savefig_kwargs
