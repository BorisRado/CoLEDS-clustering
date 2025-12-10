import torch


def compute_statistics(tensor: torch.Tensor, statistics):
    assert tensor.ndim == 2
    tensor = tensor.clone().detach()
    computed_stats = []
    if "mean" in statistics:
        tmp = tensor.mean(dim=0)
        assert tmp.shape == (tensor.shape[1], )
        computed_stats.append(tmp)
    if "q25" in statistics:
        tmp = torch.quantile(tensor, 0.25, dim=0)
        assert tmp.shape == (tensor.shape[1], )
        computed_stats.append(tmp)
    if "q50" in statistics:
        tmp = torch.quantile(tensor, 0.50, dim=0)
        assert tmp.shape == (tensor.shape[1], )
        computed_stats.append(tmp)
    if "q75" in statistics:
        tmp = torch.quantile(tensor, 0.75, dim=0)
        assert tmp.shape == (tensor.shape[1], )
        computed_stats.append(tmp)
    if "std" in statistics:
        tmp = tensor.std(dim=0)
        assert tmp.shape == (tensor.shape[1], )
        computed_stats.append(tmp)
    if "var" in statistics:
        tmp = tensor.var(dim=0)
        assert tmp.shape == (tensor.shape[1], )
        computed_stats.append(tmp)

    computed_stats = torch.cat(computed_stats)
    assert computed_stats.ndim == 1
    assert computed_stats.shape == (tensor.shape[1] * len(statistics),)
    return computed_stats.cpu().numpy()
