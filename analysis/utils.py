import os
import json
from glob import glob
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
            "max_correlation": df["correlation"].max(),
            "correlation_late_training": df["correlation"].dropna().iloc[-1],
        }
        data.append(record)
    df = pd.DataFrame(data)
    df["dataset"] = df["dataset"].map(DATASET_NAME_MAPPING)

    # sanity check
    if number_of_seeds is not None:
        assert (df.groupby("dataset").size() == number_of_seeds).all()

    df.to_csv(save_df_path, index=False)
    return df


def get_logit_dataframe(refresh=False, number_of_seeds=None):
    save_df_path = "data/all_logit_best_correlations.csv"
    if not refresh:
        try:
            return pd.read_csv(save_df_path)
        except FileNotFoundError:
            print("Could not load data. Re-downloading")
            return get_logit_dataframe(refresh=True, number_of_seeds=number_of_seeds)

    runs = get_project_runs()
    data = []
    for run in runs:
        config = run.config
        if config["profiler"] != "logit":
            continue
        df = get_run_dataframe(run)
        dataset = config["dataset.dataset_name"]
        record = {
            "dataset": dataset,
            "max_correlation": df["correlation"].max(),
            "correlation_late_training": df["correlation"].dropna().iloc[-1],
        }
        data.append(record)
    df = pd.DataFrame(data)
    df["dataset"] = df["dataset"].map(DATASET_NAME_MAPPING)

    # sanity check
    if number_of_seeds is not None:
        assert (df.groupby("dataset").size() == number_of_seeds).all()

    df.to_csv(save_df_path, index=False)
    return df


def get_pacfl_dataframe(refresh=False, number_of_seeds=None):
    save_df_path = "data/all_pacfl_best_correlations.csv"
    if not refresh:
        try:
            return pd.read_csv(save_df_path)
        except FileNotFoundError:
            print("Could not load data. Re-downloading")
            return get_pacfl_dataframe(refresh=True, number_of_seeds=number_of_seeds)

    runs = get_project_runs()
    data = []
    for run in runs:
        config = run.config
        if config["profiler"] != "pacfl":
            continue
        df = get_run_dataframe(run)
        dataset = config["dataset.dataset_name"]
        assert df["correlation"].shape[0] == 1
        record = {
            "dataset": dataset,
            "max_correlation": df["correlation"].max(),
            "correlation_late_training": df["correlation"].dropna().iloc[-1],
        }
        data.append(record)
    df = pd.DataFrame(data)
    df["dataset"] = df["dataset"].map(DATASET_NAME_MAPPING)

    # sanity check
    if number_of_seeds is not None:
        assert (df.groupby("dataset").size() == number_of_seeds).all()

    df.to_csv(save_df_path, index=False)
    return df


def add_bar_on_top(df, ax, offset, index_order):
    n_datasets = len(index_order)
    start = offset * n_datasets
    end = start + n_datasets
    wdp_bars = ax.patches[start:end]

    for bar, dataset in zip(wdp_bars, index_order):
        x = bar.get_x()
        width = bar.get_width()
        height = df.loc[dataset]["mean"]

        ax.bar(
            x,
            height,
            width=width,
            align="edge",
            color=bar.get_facecolor(),
            hatch="//",
            edgecolor="black",
            linewidth=1.0,
            zorder=bar.get_zorder() + 1,  # ensure it is drawn on top
        )

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

    cmap = sns.color_palette("colorblind", 6)
    mpl.rcParams.update({
        "figure.dpi": 200, # so it's bigger when calling plt.show()

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
        "color": [cmap[i] for i in range(6)]
    }, savefig_kwargs


def get_config_key(file, key, idx=0):
    tmp = file.split("/")[idx]
    assert tmp.find(key) >= 0, f"Key not found! {file} {key}"
    tmp = tmp[tmp.find(key) + len(key) + 1:].split("_")[0]
    return tmp


def get_num_clusters(file):
    return int(file.split("/")[1])


def get_entry(file, method, analysis_keys, compute_accuracy_fn):
    df = pd.read_csv(file)
    entry = {
        "method": method,
        "acc": compute_accuracy_fn(df),
    }
    if "coleds" in file and "save" in file:
        entry["save"] = get_config_key(file, "save")
    for key in analysis_keys:
        if key == "seed":
            entry["seed"] = int(get_config_key(file, key, 0))
        elif key == "clustering_algorithm":
            tmp_filename = file.replace("hierarchical_", "")
            entry["clustering_algorithm"] = get_config_key(tmp_filename, "clustering", idx=1)
        elif key == "ho":
            entry["num_ho_clients"] = int(get_config_key(file, "ho", idx=1))
        elif key == "mu":
            entry["algorithm"] = "FedProx" if get_config_key(file, "mu", idx=1) == "0.001" else "FedAvg"
        elif key == "num_clusters":
            try:
                num_clusters = int(file.split("/")[1])
            except:
                num_clusters = int(get_config_key(file, "clusters", 1))
            entry["num_clusters"] = num_clusters
        elif key in ["bs", "ff"]:
            entry[key] = get_config_key(file, key)
        else:
            raise Exception(f"key {key}")
    return entry


def get_accuracy_with_clustering(base_folder, data_partition, weighting_method, analysis_keys=["num_clusters", "seed"], coleds_analysis_keys=["num_clusters", "seed"]):
    assert weighting_method in {"average", "weighted_average"}
    current_directory = os.getcwd()
    os.chdir(base_folder)
    def compute_accuracy(df):
        if df["dataset_size"].sum() == 0:
            return -1
        if weighting_method == "weighted_average":
            return (df["accuracy"] * df["dataset_size"]).sum() / df["dataset_size"].sum()
        else:
            return df["accuracy"].mean()
    data = []
    try:
        filename = f"{data_partition}_accuracy.csv"
        for file in glob(f"wd_seed_*/*/{filename}"):
            data.append(
                get_entry(file, "WDP", analysis_keys, compute_accuracy)
            )

        for file in glob(f"es_*/*/{filename}"):
            method = {
                "simple": "REPA",
                "beta": "AESP"
            }[get_config_key(file, "model")]
            data.append(
                get_entry(file, method, analysis_keys, compute_accuracy)
            )

        for file in glob(f"coleds_*/*/{filename}"):
            data.append(
                get_entry(file, "CoLEDS", coleds_analysis_keys, compute_accuracy)
            )

        for file in glob(f"label*/*/{filename}"):
            data.append(
                get_entry(file, "LbP", analysis_keys, compute_accuracy)
            )
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"Exception occurred: {type(e).__name__}: {e}")
        return None
    finally:
        os.chdir(current_directory)
    return df
