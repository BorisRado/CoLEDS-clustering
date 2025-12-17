import os
import shutil
from collections import OrderedDict

import wandb
import torch
from flwr.common import parameters_to_ndarrays


def _has_converged_plateau(accuracies, patience=5):
    if len(accuracies) < patience:
        return False
    return all(x >= accuracies[-1] for x in accuracies[-patience:-1])


def get_strategy_with_chechpoint(base_strategy, file, model):
    class CustomStrategy(base_strategy):
        def __init__(self, evaluation_frequency, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert file.endswith(".pth")
            self.accuracies = []
            self.converged = False
            self.evaluation_frequency = evaluation_frequency
            self.max_accuracy = -1

        def configure_fit(self, *args, **kwargs):
            if self.converged:
                return []
            else:
                return super().configure_fit(*args, **kwargs)

        def configure_evaluate(self, server_round, parameters, client_manager):
            if self.converged or server_round % self.evaluation_frequency != 0:
                return []
            else:
                return super().configure_evaluate(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round, results, failures):
            aggregated_parameters, aggregated_metrics = \
                super().aggregate_fit(server_round, results, failures)

            print(f"Fit metrics: {aggregated_metrics}")
            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(model.state_dict(), file)

            return aggregated_parameters, aggregated_metrics

        def aggregate_evaluate(self, *args, **kwargs):
            res = super().aggregate_evaluate(*args, **kwargs)

            print("Evaluation metrics:", res[1])
            print(f"Accuracy: {res[0]}")

            if wandb.run is not None:
                wandb.log({"avg_accuracy": res[0]})
            self.accuracies.append(res[0])
            if res[0] > self.max_accuracy:
                self.max_accuracy = res[0]
                base, ext = os.path.splitext(file)
                best_file = f"{base}_best{ext}"
                shutil.copyfile(file, best_file)
                print(f"Saved new best model to {best_file}")
            return res

    return CustomStrategy
