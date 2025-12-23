# CoLEDS

This repository contains the implementation and experimental code for CoLEDS. All results presented in the paper are fully reproducible following the instructions below.

## Overview

CoLEDS is a method for dataset profiling in federated learning that uses contrastive learning to learn meaningful client representations. This repository provides the complete implementation along with scripts to reproduce all experiments from the paper.

## Requirements

### Environment Setup

1. Create a Python virtual environment and install dependencies:
   ```bash
   conda env create -f environment.yaml
   ```

2. Configure Weights & Biases for experiment tracking by creating a `.env` file in the project root:
   ```txt
   WANDB_API_KEY=<your_api_key>
   WANDB_USERNAME=<your_username>
   WANDB_ENTITY=<your_entity>
   WANDB_PROJECT=<your_project_name>
   ```

## Reproducing Experiments

### 1. CoLEDS Hyperparameter Analysis

Reproduce the hyperparameter analysis results:

1. Execute all experiments:
   ```bash
   for script in scripts/slurm/*.sh; do sbatch "$script"; done
   ```

2. Generate plots and visualizations:
   ```bash
   jupyter notebook analysis/embedding_quality.ipynb
   ```

3. Output images will be saved to `analysis/images/`

### 2. Accuracy Gains on FeMNIST

Reproduce the accuracy comparison experiments:

1. **Cache the FeMNIST dataset** (first-time setup, ~1 hour):
   ```bash
   python scripts/py/cache_femnist.py
   ```
   This creates `data/raw/femnist/` with preprocessed data for faster loading in subsequent runs.

2. **Run all accuracy experiments**:
   ```bash
   for script in scripts/slurm/accuracy_gains/*.sh; do sbatch "$script"; done
   ```

3. **Generate results table**:
   ```bash
   jupyter notebook analysis/femnist_accuracy.ipynb
   ```
   Execute all cells. The final cell outputs a LaTeX-formatted table ready for publication.

### 3. Synthetic Dataset Visualizations

Generate the synthetic dataset figures from the paper:

1. **Create dataset examples**:
   - Open `analysis/synthetic_dataset.ipynb`
   - Execute the first three cells to generate sample visualizations
   - Note: Results may vary as the RNG seed is not fixed for these visualizations

2. **Generate 2D profile representations**:
   ```bash
   sbatch scripts/slurm/synthetic/all.sh
   ```
   Then execute the remaining cells in `analysis/synthetic_dataset.ipynb` to produce the figures.

## Implementation Details

### CoLEDS Training Procedure

The core training procedure is implemented in [src/models/training_procedures.py](src/models/training_procedures.py). Two functionally equivalent implementations are provided:

- **`_optimization_iteration_sl`**: Full simulation of split learning with independent client models. Creates separate model instances for each client, simulating the complete forward-backward pass cycle. While pedagogically clear, this approach introduces overhead from model copying and management.

- **`_optimized_gradient_computation`**: Optimized implementation that achieves identical results without the overhead. Recommended for production use. This implementation is mathematically equivalent to the above but significantly faster.

**Note**: Both implementations produce identical training dynamics (loss evolution, correlation metrics). You can verify this by comparing runs with `train_config.optimized_computation=true` vs `train_config.optimized_computation=false`. For numerical precision verification, consider using double precision (`torch.float64`).

**Compatibility**: The optimized version assumes the forward pass is independent across samples. This holds for all models used in our experiments. For models with batch-dependent operations (e.g., batch normalization with `track_running_stats=True`), use the non-optimized version.

## Project Structure

```
.
├── src/                  # Source code
│   ├── models/           # Model definitions and training procedures
│   ├── clustering/       # Clustering algorithms
│   ├── data/             # Data loading utilities
│   └── utils/            # Helper functions
├── conf/                 # Hydra configuration files
├── scripts/              # Experiment scripts
│   ├── py/               # Python scripts
│   └── slurm/            # SLURM batch scripts
├── analysis/             # Jupyter notebooks for analysis and visualization
└── outputs/              # Experiment outputs and results
```

## Configuration

Experiments are configured using [Hydra](https://hydra.cc/). Configuration files are located in `conf/` and organized by:
- `dataset/`: Dataset configurations (CIFAR-10, FeMNIST, etc.)
- `model/`: Model architectures
- `optimizer/`: Optimization settings
- `partitioning/`: Data partitioning strategies

Top-level configs (e.g., `baseline.yaml`, `coleds.yaml`) define complete experiment setups.
