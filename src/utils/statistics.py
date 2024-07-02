import torch


def compute_statistic(tensor, stats, dim):
    l = []
    kwargs = {
        "input": tensor,
        "dim": dim,
        "keepdims": True
    }

    if "mean" in stats:
        l.append(torch.mean(**kwargs))
    if "quantile" in stats:
        l.append(
            torch.quantile(**kwargs, q=torch.Tensor([0.25, 0.5, 0.75])).reshape(1, -1)
        )
    if "max" in stats:
        l.append(torch.max(**kwargs).values)
    if "min" in stats:
        l.append(torch.min(**kwargs))
    if "sum" in stats:
        l.append(torch.sum(**kwargs))
    if "std" in stats:
        l.append(torch.std(**kwargs))
    if "var" in stats:
        l.append(torch.var(**kwargs))
    if "covariance" in stats:
        assert dim == 0
        cov = torch.cov(tensor.T)
        i = torch.triu_indices(*cov.shape)
        l.append(cov[i[0], i[1]].reshape(1, -1))
    if "correlation" in stats:
        assert dim == 0
        tensor_t = tensor.T
        corr = torch.corrcoef(tensor_t)
        assert corr.shape == (tensor.shape[1], tensor.shape[1])
        i = torch.triu_indices(*corr.shape, offset=1)
        corr = corr[i[0], i[1]].reshape(1, -1)
        corr = torch.nan_to_num(corr)
        l.append(corr)

    assert len(l) == len(stats), f"Some statistics are not recognized, {stats}"
    return torch.hstack(l)
