"""
MLP models.
"""

import torch.nn as nn
import torch
import numpy as np


def small_mlp_ebm(data_dim, h_dim, nout=1):
    """
    Small MLP EBM.
    """
    return nn.Sequential(
        nn.Linear(data_dim, h_dim),
        #nn.LeakyReLU(.2, inplace=True),
        nn.PReLU(),
        nn.Linear(h_dim, h_dim),
        #nn.LeakyReLU(.2, inplace=True),
        nn.PReLU(),
        nn.Linear(h_dim, nout, bias=True)
    )
def small_mlp_ebm_sn(data_dim, h_dim, nout=1):
    """
    Small MLP EBM.
    """
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Linear(data_dim, h_dim)),
        #nn.LeakyReLU(.2, inplace=True),
        nn.PReLU(),
        nn.utils.spectral_norm(nn.Linear(h_dim, h_dim)),
        #nn.LeakyReLU(.2, inplace=True),
        nn.PReLU(),
        nn.utils.spectral_norm(nn.Linear(h_dim, nout, bias=True))
    )

def linear_generator(noise_dim, data_dim):
    """
    Linear generator.
    """
    return nn.Linear(noise_dim, data_dim, bias=False)


def small_mlp_generator(noise_dim, data_dim, h_dim):
    """
    Small MLP generator.
    """
    return nn.Sequential(
        nn.Linear(noise_dim, h_dim, bias=True),
        nn.PReLU(),
        nn.BatchNorm1d(h_dim, affine=True),
        #nn.ReLU(inplace=True),
        nn.Linear(h_dim, h_dim, bias=True),
        nn.PReLU(),
        nn.BatchNorm1d(h_dim, affine=True),
        #nn.ReLU(inplace=True),
        nn.Linear(h_dim, data_dim, bias=True)
    )

def small_mlp_generator_no_bn(noise_dim, data_dim, h_dim):
    """
    Small MLP generator.
    """
    return nn.Sequential(
        nn.Linear(noise_dim, h_dim),
        nn.ReLU(inplace=True),
        nn.Linear(h_dim, h_dim),
        nn.ReLU(inplace=True),
        nn.Linear(h_dim, data_dim)
    )


def large_mlp_ebm(data_dim, nout=1, weight_norm=True):
    """
    Large MLP EBM.
    """
    if weight_norm:
        return nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.utils.weight_norm(nn.Linear(data_dim, 1000)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(1000, 500)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(500, 500)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(500, 250)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(250, 250)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(250, 250)),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(250, nout, bias=True)
        )
    else:
        return nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(data_dim, 1000),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(1000, 500),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(500, 500),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(500, 250),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(250, 250),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(250, 250),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(250, nout, bias=True)
        )


def large_mlp_generator(noise_dim, data_dim, unit_interval, no_final_act=False):
    """
    Large MLP generator.
    """
    if no_final_act:
        final_act = nn.Identity()
    else:
        if unit_interval:
            final_act = nn.Sigmoid()
        else:
            final_act = nn.Tanh()
    return nn.Sequential(
        nn.Linear(noise_dim, 500, bias=False),
        nn.BatchNorm1d(500, affine=True),
        nn.Softplus(),
        nn.Linear(500, 500, bias=False),
        nn.BatchNorm1d(500, affine=True),
        nn.Softplus(),
        nn.Linear(500, 500, bias=False),
        nn.BatchNorm1d(500, affine=True),
        nn.Softplus(),
        nn.Linear(500, data_dim, bias=True),
        final_act
    )


class MOG(nn.Module):
    def __init__(self, data_dim, n_comps):
        super().__init__()
        self._init = True
        self.mu = nn.Parameter(torch.randn((n_comps, data_dim)))
        self.logstd = nn.Parameter(torch.zeros((n_comps, data_dim)))
        self.logits = nn.Parameter(torch.zeros((n_comps,)))
        self.n_comps = n_comps
        self.data_dim = data_dim

    def forward(self, x):
        if not self._init:
            self.mu.data = x[:self.n_comps]
            self._init = True
        x = x.view(x.size(0), -1)
        dists = torch.distributions.Normal(self.mu[None], self.logstd.exp()[None])
        lps = dists.log_prob(x[:, None, :]).sum(-1)
        log_pi = self.logits.log_softmax(0)
        lps = lps + log_pi[None]
        lps = lps.logsumexp(-1)
        return lps

    def sample(self, n):
        pi = torch.distributions.OneHotCategorical(logits=self.logits)
        inds = pi.sample((n,))
        eps = torch.randn((n, self.n_comps, self.data_dim)).to(self.mu.device)
        x_all = eps * self.logstd.exp()[None] + self.mu[None]
        x = (x_all * inds[:, :, None]).sum(1)
        return x


class NICE(nn.Module):
    def __init__(self, size, hidden_nodes, num_layers=2):
        super().__init__()
        self.nice1 = NiceLayer(size, hidden_nodes, num_layers)
        self.nice2 = NiceLayer(size, hidden_nodes, num_layers)
        self.nice3 = NiceLayer(size, hidden_nodes, num_layers)
        self.nice4 = NiceLayer(size, hidden_nodes, num_layers)
        self.scale = NiceScaleLayer(size)
        self.size = size

    def _permutate(self, tensor, neurons, inv=False):
        permutation = np.arange(0, neurons)
        perm = permutation.copy()
        perm[:len(permutation) // 2] = permutation[::2]
        perm[len(permutation) // 2:] = permutation[1::2]
        inv_perm = np.argsort(perm)
        if not inv:
            to_perm = torch.from_numpy(np.identity(len(permutation))[:, perm]).to(tensor.device).type_as(tensor)
            return tensor @ to_perm
        else:
            inv_perm = torch.from_numpy(np.identity(len(permutation))[:, inv_perm]).to(tensor.device).type_as(tensor)
            return tensor @ inv_perm

    def forward(self, X, inv=False, return_y=False):
        if not inv:
            X = X.view(X.size(0), -1)
            y = self._permutate(X, X.shape[1], inv=inv)
            jac_pre = 0.0
            y = self.nice1(y, type=0, inv=inv)
            y = self.nice2(y, type=1, inv=inv)
            y = self.nice3(y, type=0, inv=inv)
            y = self.nice4(y, type=1, inv=inv)

            y, jac = self.scale(y, inv=inv)
            dim = y.shape[1]
            self.output = y
            if return_y:
                return -torch.tensor(dim * 0.5 * np.log(2 * np.pi), device=X.device) \
                       - 0.5 * torch.sum(y ** 2, dim=1) + jac + jac_pre, y
            else:
                return -torch.tensor(dim * 0.5 * np.log(2 * np.pi), device=X.device) \
                       - 0.5 * torch.sum(y ** 2, dim=1) + jac + jac_pre
        else:
            y, jac = self.scale(X, inv=inv)
            dim = y.shape[1]
            y = self.nice4(y, type=1, inv=inv)
            y = self.nice3(y, type=0, inv=inv)
            y = self.nice2(y, type=1, inv=inv)
            y = self.nice1(y, type=0, inv=inv)
            return self._permutate(y, X.shape[1], inv=inv), \
                   -torch.tensor(dim * 0.5 * np.log(2 * np.pi), device=X.device) - 0.5 * torch.sum(X ** 2, dim=1) - jac

    def inv_scale_jac(self):
        return self.scale.inv_scale_jac()

    def sample(self, n):
        y = torch.randn((n, self.size)).to(self.scale.scales.device)
        x, _ = self.forward(y, inv=True)
        return x


class NiceLayer(nn.Module):
    # Note: only support num_layers=2, with tanh (as in OLDNICE)
    # or num_layers=5 with relu (as in NICEPAPER)
    def __init__(self, size, hidden_size, num_layers=2):
        super().__init__()
        self.half_size = half_size = size // 2
        self.num_layers = num_layers
        self.dense1 = nn.Linear(half_size, hidden_size)
        self.act1 = nn.Softplus()
        if self.num_layers == 2:
            self.dense2 = nn.Linear(hidden_size, half_size)

        elif self.num_layers == 5:
            self.dense2 = nn.Linear(hidden_size, hidden_size)
            self.act2 = nn.Softplus()
            self.dense3 = nn.Linear(hidden_size, hidden_size)
            self.act3 = nn.Softplus()
            self.dense4 = nn.Linear(hidden_size, hidden_size)
            self.act4 = nn.Softplus()
            self.dense5 = nn.Linear(hidden_size, half_size)
        else:
            raise ValueError("Only supports 2 or 5 layers in a coupling layer")

    def _m_net(self, X):
        if self.num_layers == 2:
            l1 = self.act1(self.dense1(X))
            return self.dense2(l1)
        else:
            l1 = self.act1(self.dense1(X))
            l2 = self.act2(self.dense2(l1))
            l3 = self.act3(self.dense3(l2))
            l4 = self.act4(self.dense4(l3))
            l5 = self.dense5(l4)
            return l5


    def forward(self, X, type=0, inv=False):
        x1 = X[:, :self.half_size]
        x2 = X[:, self.half_size:]
        if type == 0:
            m1 = self._m_net(x1)
            delta = torch.cat([torch.zeros_like(x1), m1], dim=1)
        elif type == 1:
            m2 = self._m_net(x2)
            delta = torch.cat([m2, torch.zeros_like(x2)], dim=1)

        if not inv:
            return X + delta
        else:
            return X - delta


class NiceScaleLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.scales = nn.Parameter(torch.zeros(1, size))

    def forward(self, X, inv=False):
        if not inv:
            self.grad1 = torch.exp(self.scales)
            self.grad2 = 0.

            return X * torch.exp(self.scales), torch.sum(self.scales)
        else:
            self.grad1 = self.grad2 = None
            return X * torch.exp(-self.scales), torch.sum(-self.scales)

    def inv_scale_jac(self):
        return torch.sum(-self.scales)









if __name__ == "__main__":
    mog = MOG(2, 10)
    x = torch.randn((13, 2))
    lp = mog(x)

    mog.sample(13)

    nice = NICE(784, 1000, 5)
    x = torch.randn((13, 784))

    lpx, z = nice(x, return_y=True)
    print(lpx.size(), z.size())

    xr, lpz = nice(z, inv=True)

    print(xr.size(), lpz.size())

    print((x - xr).abs().mean())
