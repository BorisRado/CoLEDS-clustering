from functools import partial
from multiprocessing import Pool

import datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner


def _load_partition(idx, fds):
    fds.load_partition(idx)
    # print(f"Loaded {idx}", flush=True)


def run():
    import os
    tmp = os.getenv("HF_DATASETS_CACHE")
    print(tmp)

    partitioner = NaturalIdPartitioner(partition_by="writer_id")
    fds = FederatedDataset(dataset="flwrlabs/femnist", partitioners={"train": partitioner})
    func_load_partition = partial(_load_partition, fds=fds)
    datasets.disable_progress_bars()
    femnistmy_datasets = [fds.load_partition(idx) for idx in range(3597)]
    # with Pool(processes=32) as pool:
    #     pool.map(func_load_partition, range(3597))


if __name__ == "__main__":
    run()
