from pathlib import Path

import hydra
import torch


def get_exp_folder():
    exp_folder = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print("Experiment folder: ", exp_folder)
    return exp_folder


def set_torch_flags():
    # these flags help us to achieve better numerical stability across experiments and hence
    # help achieving more reproducible results
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_flush_denormal(True)
