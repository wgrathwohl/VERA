"""
Visualization utilities.
"""

import numpy as np
import torch


def plt_toy_density(logdensity, ax, npts=100,
                    title="$q(x)$", device="cpu", low=-4, high=4, exp=True):
    """
    Plot density of toy data.
    """
    side = np.linspace(low, high, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)

    logpx = logdensity(x).squeeze()

    if exp:
        logpx = logpx - logpx.logsumexp(0)
        px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
        px = px / px.sum()
    else:
        logpx = logpx - logpx.logsumexp(0)
        px = logpx.cpu().detach().numpy().reshape(npts, npts)

    ax.imshow(px)
    ax.set_title(title)
