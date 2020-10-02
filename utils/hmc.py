"""
Utilities for performing HMC sampling.
TODO: get rid of duplicate code
"""

import torch
import torch.distributions as distributions


def _gen_post_helper(netG, x_tilde, eps, sigma):
    eps = eps.clone().detach().requires_grad_(True)
    with torch.no_grad():
        G_eps = netG(eps)
    bsz = eps.size(0)
    log_prob_eps = (eps ** 2).view(bsz, -1).sum(1).view(-1, 1)
    log_prob_x = (x_tilde - G_eps)**2 / sigma**2
    log_prob_x = log_prob_x.view(bsz, -1)
    log_prob_x = torch.sum(log_prob_x, dim=1).view(-1, 1)
    logjoint_vect = -0.5 * (log_prob_eps + log_prob_x)
    logjoint_vect = logjoint_vect.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint = eps.grad
    return logjoint_vect, logjoint, grad_logjoint


def get_gen_posterior_samples(netG, x_tilde, eps_init, sigma, burn_in, num_samples_posterior,
            leapfrog_steps, stepsize, flag_adapt, hmc_learning_rate, hmc_opt_accept):
    device = eps_init.device
    bsz, eps_dim = eps_init.size(0), eps_init.size(1)
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)
    samples = torch.zeros(bsz*num_samples_posterior, eps_dim).to(device)
    current_eps = eps_init
    cnt = 0
    for i in range(n_steps):
        eps = current_eps
        p = torch.randn_like(current_eps)
        current_p = p
        logjoint_vect, logjoint, grad_logjoint = _gen_post_helper(netG, x_tilde, current_eps, sigma)
        current_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        for j in range(leapfrog_steps):
            eps = eps + stepsize * p
            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, grad_logjoint = _gen_post_helper(netG, x_tilde, eps, sigma)
                proposed_U = -logjoint_vect
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U
        logjoint_vect, logjoint, grad_logjoint = _gen_post_helper(netG, x_tilde, eps, sigma)
        proposed_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        p = -p
        current_K = 0.5 * (current_p**2).sum(dim=1)
        current_K = current_K.view(-1, 1)       # should be size of B x 1
        proposed_K = 0.5 * (p**2).sum(dim=1)
        proposed_K = proposed_K.view(-1, 1)     # should be size of B x 1
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U - proposed_U + current_K - proposed_K))
        accept = accept.float().squeeze()       # should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try:
            len(ind) > 0
            current_eps[ind, :] = eps[ind, :]
            current_U[ind] = proposed_U[ind]
        except:
            print('Samples were all rejected...skipping')
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = stepsize + hmc_learning_rate * (accept.float().mean() - hmc_opt_accept) * stepsize
        else:
            if eps_dim == 1:
                samples[cnt*bsz: (cnt+1)*bsz, :] = current_eps
            else:
                samples[cnt*bsz: (cnt+1)*bsz, :] = current_eps.squeeze()
            cnt += 1

        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples, acceptRate, stepsize


def _ebm_helper(netEBM, x):
    x = x.clone().detach().requires_grad_(True)
    E_x = netEBM(x)
    logjoint_vect = E_x.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint = x.grad
    return logjoint_vect, logjoint, grad_logjoint


def get_ebm_samples(netEBM, x_init, burn_in, num_samples_posterior,
                    leapfrog_steps, stepsize, flag_adapt, hmc_learning_rate, hmc_opt_accept):
    device = x_init.device
    bsz, x_size = x_init.size(0), x_init.size()[1:]
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)
    samples = torch.zeros(bsz*num_samples_posterior, *x_size).to(device)
    current_x = x_init
    cnt = 0
    for i in range(n_steps):
        x = current_x
        p = torch.randn_like(current_x)
        current_p = p
        logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, current_x)
        current_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        for j in range(leapfrog_steps):
            x = x + stepsize * p
            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, x)
                proposed_U = -logjoint_vect
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U
        logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, x)
        proposed_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        p = -p
        current_K = 0.5 * (current_p**2).flatten(start_dim=1).sum(dim=1)
        current_K = current_K.view(-1, 1)       # should be size of B x 1
        proposed_K = 0.5 * (p**2).flatten(start_dim=1).sum(dim=1)
        proposed_K = proposed_K.view(-1, 1)     # should be size of B x 1
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U - proposed_U + current_K - proposed_K))
        accept = accept.float().squeeze()       # should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try:
            len(ind) > 0
            current_x[ind, :] = x[ind, :]
            current_U[ind] = proposed_U[ind]
        except:
            print('Samples were all rejected...skipping')
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = stepsize + hmc_learning_rate * (accept.float().mean() - hmc_opt_accept) * stepsize
        else:
            samples[cnt*bsz: (cnt+1)*bsz, :] = current_x
            cnt += 1
        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples, acceptRate, stepsize


def _ebm_latent_helper(netEBM, netG, z, eps, sigma):
    z = z.clone().detach().requires_grad_(True)
    eps = eps.clone().detach().requires_grad_(True)
    x = netG(z) + eps * sigma
    E_x = netEBM(x)
    logjoint_vect = E_x.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint_z = z.grad
    grad_logjoint_eps = eps.grad
    return logjoint_vect, logjoint, (grad_logjoint_z, grad_logjoint_eps)


def get_ebm_latent_samples(netEBM, netG, z_init, eps_init, sigma,
                           burn_in, num_samples_posterior, leapfrog_steps, stepsize,
                           flag_adapt, hmc_learning_rate, hmc_opt_accept):
    device = z_init.device
    bsz, z_size, eps_size = z_init.size(0), z_init.size()[1:], eps_init.size()[1:]
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)

    samples_z = torch.zeros(bsz*num_samples_posterior, *z_size).to(device)
    samples_eps = torch.zeros(bsz*num_samples_posterior, *eps_size).to(device)

    current_z = z_init
    current_eps = eps_init
    cnt = 0
    for i in range(n_steps):
        z = current_z
        eps = current_eps

        p_z = torch.randn_like(current_z)
        p_eps = torch.randn_like(current_eps)

        current_p_z = p_z
        current_p_eps = p_eps

        logjoint_vect, logjoint, (grad_logjoint_z, grad_logjoint_eps) = _ebm_latent_helper(netEBM, netG, current_z, current_eps, sigma)
        current_U = -logjoint_vect.view(-1, 1)

        grad_U_z = -grad_logjoint_z
        grad_U_eps = -grad_logjoint_eps

        p_z = p_z - stepsize * grad_U_z / 2.0
        p_eps = p_eps - stepsize * grad_U_eps / 2.0

        for j in range(leapfrog_steps):

            z = z + stepsize * p_z
            eps = eps + stepsize * p_eps

            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, (grad_logjoint_z, grad_logjoint_eps) = \
                    _ebm_latent_helper(netEBM, netG, z, eps, sigma)
                proposed_U = -logjoint_vect

                grad_U_z = -grad_logjoint_z
                grad_U_eps = -grad_logjoint_eps

                p_z = p_z - stepsize * grad_U_z
                p_eps = p_eps - stepsize * grad_U_eps

        logjoint_vect, logjoint, (grad_logjoint_z, grad_logjoint_eps) = _ebm_latent_helper(netEBM, netG, z, eps, sigma)
        proposed_U = -logjoint_vect.view(-1, 1)

        grad_U_z = -grad_logjoint_z
        grad_U_eps = -grad_logjoint_eps

        p_z = p_z - stepsize * grad_U_z / 2.0
        p_z = -p_z

        p_eps = p_eps - stepsize * grad_U_eps / 2.0
        p_eps = -p_eps

        current_K = 0.5 * (current_p_z**2).flatten(start_dim=1).sum(dim=1)
        current_K += 0.5 * (current_p_eps**2).flatten(start_dim=1).sum(dim=1)
        current_K = current_K.view(-1, 1)       # should be size of B x 1

        proposed_K = 0.5 * (p_z**2).flatten(start_dim=1).sum(dim=1)
        proposed_K += 0.5 * (p_eps ** 2).flatten(start_dim=1).sum(dim=1)
        proposed_K = proposed_K.view(-1, 1)     # should be size of B x 1
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U - proposed_U + current_K - proposed_K))
        accept = accept.float().squeeze()       # should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try:
            len(ind) > 0
            current_z[ind, :] = z[ind, :]
            current_eps[ind, :] = eps[ind, :]
            current_U[ind] = proposed_U[ind]
        except:
            print('Samples were all rejected...skipping')
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = stepsize + hmc_learning_rate * (accept.float().mean() - hmc_opt_accept) * stepsize
        else:
            samples_z[cnt*bsz: (cnt+1)*bsz, :] = current_z.squeeze()
            samples_eps[cnt * bsz: (cnt + 1) * bsz, :] = current_eps.squeeze()
            cnt += 1
        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples_z, samples_eps, acceptRate, stepsize


def sgld_sample(logp_fn, x_init, l=1., e=.01, n_steps=100):
    x_k = torch.autograd.Variable(x_init, requires_grad=True)
    # sgld
    lrs = [l for _ in range(n_steps)]
    for this_lr in lrs:
        f_prime = torch.autograd.grad(logp_fn(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += this_lr * f_prime + torch.randn_like(x_k) * e
    final_samples = x_k.detach()
    return final_samples


def update_logp(u, u_mu, std):
    return distributions.Normal(u_mu, std).log_prob(u).flatten(start_dim=1).sum(1)


def MALA(vars, logp_fn, step_lr):
    """
    Metropolis-Adjusted Langevin Algorithm.
    """
    step_std = (2 * step_lr) ** .5
    logp_vars = logp_fn(*vars)
    grads = torch.autograd.grad(logp_vars.sum(), vars)
    updates_mu = [v + step_lr * g for v, g in zip(vars, grads)]
    updates = [u_mu + step_std * torch.randn_like(u_mu) for u_mu in updates_mu]
    logp_updates = logp_fn(*updates)
    reverse_grads = torch.autograd.grad(logp_updates.sum(), updates)
    reverse_updates_mu = [v + step_lr * g for v, g in zip(updates, reverse_grads)]

    logp_forward = sum([update_logp(u, u_mu, step_std) for u, u_mu in zip(updates, updates_mu)])
    logp_backward = sum([update_logp(v, ru_mu, step_std) for v, ru_mu in zip(vars, reverse_updates_mu)])
    logp_accept = logp_updates + logp_backward - logp_vars - logp_forward
    p_accept = logp_accept.exp()
    accept = (torch.rand_like(p_accept) < p_accept).float()

    next_vars = []
    for u_v, v in zip(updates, vars):
        if len(u_v.size()) == 4:
            next_vars.append(accept[:, None, None, None] * u_v + (1 - accept[:, None, None, None]) * v)
        else:
            next_vars.append(accept[:, None] * u_v + (1 - accept[:, None]) * v)
    return next_vars, accept.mean()
