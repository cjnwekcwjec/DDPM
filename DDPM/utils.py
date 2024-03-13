import torch


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6]
    t = [1, 3]
    x = torch.tensor(x)
    t = torch.tensor(t)
    y = gather(x, t)
    print(y)
    print(y.shape)

