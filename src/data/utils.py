import torch


def tensor2numpy(x):
    try:
        return x.cpu().numpy()
    except TypeError:
        return x.detach().cpu().numpy()


def numpy2tensor(x):
    return torch.Tensor(x)
