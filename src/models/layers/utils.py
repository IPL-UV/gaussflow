import torch
import FrEIA.modules as Fm


class NFlowsLayer(Fm.InvertibleModule):
    def __init__(
        self,
        dims_in,
        transform,
        name="linear",
    ):
        super().__init__(dims_in)

        self.n_features = dims_in[0][0]
        self.transform = transform
        self.name = name

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        if rev:

            z, log_det = self.transform.inverse(x)
            # print(f"Mix (Out): {z.min(), z.max()}")

        else:
            # print(x.shape)
            z, log_det = self.transform.forward(x)

        return (z,), log_det

    def output_dims(self, input_dims):
        return input_dims


def construct_householder_matrix(V):
    n_reflections, n_channels = V.shape

    I = torch.eye(n_channels, dtype=V.dtype, device=V.device)

    Q = I

    for i in range(n_reflections):
        v = V[i].view(n_channels, 1)

        vvT = torch.matmul(v, v.t())
        vTv = torch.matmul(v.t(), v)
        Q = torch.matmul(Q, I - 2 * vvT / vTv)

    return Q


def bisection_inverse(fn, z, init_x, init_lower, init_upper, eps=1e-10, max_iters=100):
    """Bisection method to find the inverse of `fn`. Computed by finding the root of `z-fn(x)=0`."""

    def body(x_, lb_, ub_, cur_z_):
        gt = (cur_z_ > z).type(z.dtype)
        lt = 1 - gt
        new_x_ = gt * (x_ + lb_) / 2.0 + lt * (x_ + ub_) / 2.0
        new_lb = gt * lb_ + lt * x_
        new_ub = gt * x_ + lt * ub_
        return new_x_, new_lb, new_ub

    x, lb, ub = init_x, init_lower, init_upper
    cur_z = fn(x)
    diff = float("inf")
    i = 0
    while diff > eps and i < max_iters:
        x, lb, ub = body(x, lb, ub, cur_z)
        cur_z = fn(x)
        diff = (z - cur_z).abs().max()
        i += 1

    return x


def share_across_batch(params, batch_size):
    return params[None, ...].expand(batch_size, *params.shape)


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""

    pass


class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""

    pass