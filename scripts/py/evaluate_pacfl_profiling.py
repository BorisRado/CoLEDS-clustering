import uuid
import copy
from dotenv import load_dotenv

import torch
import hydra
import wandb
import numpy as np
from hydra.core.config_store import OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.other import set_torch_flags
from src.data.utils import get_datasets_from_cfg
from src.utils.stochasticity import set_seed
from src.utils.wandb import init_wandb, finish_wandb


def _compute_svd_bases_for_clients(
    valsets,
    num_basis: int,
):
    bases = []
    for ds in valsets:
        X = ds.tensors[0].flatten(start_dim=1).T.numpy()

        u1_temp, _, _ = np.linalg.svd(X, full_matrices=False)
        u1_temp=u1_temp / np.linalg.norm(u1_temp, ord=2, axis=0)
        u1_temp = u1_temp[:, 0:num_basis]

        bases.append(u1_temp)
    return bases


def calculating_adjacency(U):

    # copy-pasted from the original implementation
    nclients = len(U)
    clients_idxs = np.arange(nclients)

    sim_mat = np.zeros([nclients, nclients])
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])

            mul = np.clip(U1.T@U2 ,a_min =-1.0, a_max=1.0)
            similarity = np.min(np.arccos(mul))*180/np.pi
            sim_mat[idx1,idx2] = similarity
            sim_mat[idx2,idx1] = similarity

    return sim_mat

def calculating_adjacency_original(clients_idxs, U):

    nclients = len(clients_idxs)

    sim_mat = np.zeros([nclients, nclients])
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            #print(idx1)
            #print(U)
            #print(idx1)
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])

            #sim_mat[idx1,idx2] = np.where(np.abs(U1.T@U2) > 1e-2)[0].shape[0]
            #sim_mat[idx1,idx2] = 10*np.linalg.norm(U1.T@U2 - np.eye(15), ord='fro')
            #sim_mat[idx1,idx2] = 100/np.pi*(np.sort(np.arccos(U1.T@U2).reshape(-1))[0:4]).sum()
            mul = np.clip(U1.T@U2 ,a_min =-1.0, a_max=1.0)
            sim_mat[idx1,idx2] = np.min(np.arccos(mul))*180/np.pi

    return sim_mat



@hydra.main(version_base=None, config_path="../../conf", config_name="pacfl")
def run(cfg):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    if cfg.get("dry_run", False) is True:
        return
    init_wandb(cfg)

    set_seed(cfg.general.seed)
    _, valsets = get_datasets_from_cfg(cfg)

    ### This is a copy-paste from the official implementation
    # K = cfg.profiling.num_basis
    # U_clients = []
    # for idx, dataset in enumerate(valsets):

    #     idxs_local = np.arange(len(dataset))
    #     labels_local = dataset.tensors[1].numpy()
    #     # Sort Labels Train
    #     idxs_labels_local = np.vstack((idxs_local, labels_local))
    #     idxs_labels_local = idxs_labels_local[:, idxs_labels_local[1, :].argsort()]
    #     idxs_local = idxs_labels_local[0, :]
    #     labels_local = idxs_labels_local[1, :]

    #     uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)

    #     print(f'Labels: {uni_labels}, Counts: {cnt_labels}')

    #     nlabels = len(uni_labels)
    #     cnt = 0
    #     U_temp = []
    #     for j in range(nlabels):
    #         local_ds1 = dataset.tensors[0][idxs_local[cnt:cnt+cnt_labels[j]]]
    #         local_ds1 = local_ds1.reshape(cnt_labels[j], -1)
    #         local_ds1 = local_ds1.T

    #         u1_temp, sh1_temp, vh1_temp = np.linalg.svd(local_ds1, full_matrices=False)
    #         u1_temp = u1_temp / np.linalg.norm(u1_temp, ord=2, axis=0)
    #         U_temp.append(u1_temp[:, 0:K])

    #         cnt+=cnt_labels[j]
    #     U_clients.append(copy.deepcopy(np.hstack(U_temp)))

    #     print(f'Shape of U: {U_clients[-1].shape}')

    # clients_idxs = np.arange(len(valsets))
    # svd_similarity_matrix = calculating_adjacency_original(clients_idxs, U_clients)

    # Construct label similarity matrix
    label_distributions = []
    for valset in valsets:
        label_dist = valset._label_distribution.reshape(1, -1)
        label_distributions.append(label_dist)
    label_distributions = np.vstack(label_distributions)
    label_distribution_similarity = cosine_similarity(label_distributions)

    # Compute SVD-based subspace bases
    bases = _compute_svd_bases_for_clients(
        valsets=valsets,
        num_basis=cfg.profiling.num_basis,
    )

    # Compute the similarity matrix between clients (PACFL-style)
    svd_similarity_matrix = -calculating_adjacency(bases)

    print("PACFL SVD-based client similarity matrix:")
    print(svd_similarity_matrix)

    sim1 = label_distribution_similarity[np.triu_indices(label_distribution_similarity.shape[0], 1)]
    sim2 = svd_similarity_matrix[np.triu_indices(svd_similarity_matrix.shape[0], 1)]

    correlation = np.corrcoef(sim1, sim2)[0, 1]
    print(correlation)
    print("=======")
    if wandb.run is not None:
        wandb.log({"correlation": correlation})
    finish_wandb()


if __name__ == "__main__":
    OmegaConf.register_new_resolver("uuid", lambda: str(uuid.uuid4())[:8])
    set_torch_flags()
    load_dotenv()
    run()
