"""
Code for the VAT baseline.
"""

import torch
import torch.nn as nn


def _l2_normalize(d):
    """
    L2 normalization for VAT.
    """
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    """
    The loss for VAT.
    """
    # Adapted, with permission, from https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py

    def __init__(self, xi=10.0, eps=2.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = torch.nn.functional.softmax(model.classify(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            pred_hat = model.classify(x + self.xi * d)
            logp_hat = torch.nn.functional.log_softmax(pred_hat, dim=1)
            adv_distance = torch.nn.functional.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            model.zero_grad()

        # calc LDS
        r_adv = d * self.eps
        pred_hat = model.classify(x + r_adv)
        logp_hat = torch.nn.functional.log_softmax(pred_hat, dim=1)
        lds = torch.nn.functional.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
