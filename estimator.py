"""
Estimate the bias and variance of the entropy estimator.
"""

import argparse
from math import floor, ceil

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

import utils
from models import linear_generator, VERAHMCGenerator, VERAGenerator


def main(args):
    """
    The main function.
    """
    torch.manual_seed(args.seed)
    cpu_device = torch.device('cpu')

    if args.dataset == "mnist":
        data_size = (1, 28, 28)
        noise_dim = 100
        data_dim = 784

        # put data in [-1, 1] and flatten
        post_trans = [lambda x: 2 * x - 1]
        post_trans_inv = lambda x: (x + 1) / 2

        pre_trans = [transforms.ToTensor()]
        post_trans += [lambda x: x.view(-1)]

        if not (args.load_pca or args.load_pca_params):
            tr_dataset = datasets.MNIST("./data",
                                        transform=transforms.Compose(pre_trans + post_trans),
                                        download=True)
            tr_dload = iter(DataLoader(tr_dataset, len(tr_dataset), True, drop_last=True))
            x_d, _ = next(tr_dload)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))
    else:
        noise_dim = 1
        data_dim = 2
        x_d = torch.tensor(utils.toy_data.inf_train_gen("moons", batch_size=args.batch_size)).float()

    if args.load_pca_params:
        with open("pca_params.pickle", "rb") as pickle_file:
            pca_params = pickle.load(pickle_file)
        loc = pca_params["loc"]
        components = pca_params["components"]
        exp_var_diff = pca_params["exp_var_diff"]
        cov_factor = pca_params["cov_factor"]
        cov_diag = pca_params["cov_diag"]
    else:
        if args.load_pca:
            with open("pca.pickle", "rb") as pickle_file:
                transformer = pickle.load(pickle_file)
        else:
            transformer = PCA(n_components=noise_dim, random_state=args.seed)
            transformer.fit_transform(x_d.detach().cpu().numpy())

        with open("pca.pickle", "wb") as pickle_file:
            pickle.dump(transformer, pickle_file)

        # put parameters into torch normal distribution
        loc = torch.tensor(transformer.mean_)[None]                                     # (1, data_dim)
        components = torch.tensor(transformer.components_)
        exp_var_diff = torch.tensor(np.maximum(transformer.explained_variance_ - transformer.noise_variance_, 0.))
        cov_factor = (components.T * exp_var_diff.sqrt())[None]                         # (1, data_dim, noise_dim)
        cov_diag = torch.tensor(transformer.noise_variance_) * torch.ones_like(loc)     # (1, data_dim)

        pca_params = {
            "loc": loc,
            "components": components,
            "exp_var_diff": exp_var_diff,
            "cov_factor": cov_factor,
            "cov_diag": cov_diag
        }

        with open("pca_params.pickle", "wb") as pickle_file:
            pickle.dump(pca_params, pickle_file)

    generator_net = linear_generator(noise_dim=noise_dim, data_dim=data_dim)
    if args.estimator == "hmc":
        generator = VERAHMCGenerator(generator_net, noise_dim=noise_dim, mcmc_lr=.01)
    else:
        generator = VERAGenerator(generator_net, noise_dim=noise_dim, post_lr=.01)

    class LinearGenerator(nn.Module):
        """
        Defines distribution of linear gen of latent gaussian variables.
        """
        def __init__(self, gen):
            super().__init__()
            self.gen = gen

        def dist(self, device):
            """
            The distribution induced by the gen.
            """
            W = self.gen.g.weight.data
            WtW = W @ W.t()
            cov = WtW + torch.eye(WtW.size(0)).to(device) * self.gen.logsigma.exp() ** 2
            mu = self.gen.g.bias
            return MultivariateNormal(mu, cov)

    # get samples directly from gen and from distribution induced by gen
    generator.to(device)
    lgen_inf = LinearGenerator(generator)
    lgen_inf.to(device)

    if args.check_dist_eq:
        dist = lgen_inf.dist(cpu_device)

        dist_samples = dist.rsample((args.batch_size,))
        x_g, _ = generator.sample(args.batch_size, requires_grad=False)

        # print statistics of samples to confirm they are the same when using random parameters
        utils.print_log("BEFORE", args)
        utils.print_log(f"Absolute difference in mean | "
                        f"max: {(x_g.mean(0) - dist_samples.mean(0)).abs().max().item():.3f} | "
                        f"min: {(x_g.mean(0) - dist_samples.mean(0)).abs().min().item():.3f} | "
                        f"mean: {(x_g.mean(0) - dist_samples.mean(0)).abs().mean().item():.3f}", args)

        utils.print_log(f"Absolute difference in std | "
                        f"max: {(x_g.std(0) - dist_samples.std(0)).abs().max().item():.3f} | "
                        f"min: {(x_g.std(0) - dist_samples.std(0)).abs().min().item():.3f} | "
                        f"mean: {(x_g.std(0) - dist_samples.std(0)).abs().mean().item():.3f}", args)

    def set_gen_params(gen):
        """
        Set the params of the linear gen.
        """
        state_dict = gen.state_dict()

        state_dict["g.bias"] = loc[0]
        state_dict["g.weight"] = cov_factor[0]
        state_dict["logsigma"] = (cov_diag[0, 0] * torch.ones((1, ))).log()

        gen.load_state_dict(state_dict)

    # set parameters so that they induce the same distribution given by PPCA
    utils.print_log("setting params", args)
    set_gen_params(generator)

    if args.check_dist_eq:
        dist = lgen_inf.dist(cpu_device)  # reinitialize the distribution using the new params pointed to be lgen_inf

        dist_samples = dist.rsample((args.batch_size,))
        x_g, _ = generator.sample(args.batch_size, requires_grad=False)

        # print statistics of samples to confirm they are the same when using parameters given by PCA
        utils.print_log("AFTER", args)
        utils.print_log(f"Absolute difference in mean | "
                        f"max: {(x_g.mean(0) - dist_samples.mean(0)).abs().max().item():.3f} | "
                        f"min: {(x_g.mean(0) - dist_samples.mean(0)).abs().min().item():.3f} | "
                        f"mean: {(x_g.mean(0) - dist_samples.mean(0)).abs().mean().item():.3f}", args)

        utils.print_log(f"Absolute difference in std | "
                        f"max: {(x_g.std(0) - dist_samples.std(0)).abs().max().item():.3f} | "
                        f"min: {(x_g.std(0) - dist_samples.std(0)).abs().min().item():.3f} | "
                        f"mean: {(x_g.std(0) - dist_samples.std(0)).abs().mean().item():.3f}", args)

        if args.plot_samples:
            if args.dataset == "mnist":
                plot("dist.png", dist_samples.view(dist_samples.size(0), *data_size)[:100])
                plot("g.png", x_g.view(x_g.size(0), *data_size)[:100])
            else:
                # plot samples from distribution and gen
                plt.clf()
                plt.xlim([-8, 8])
                plt.ylim([-8, 8])
                plt.scatter(x_d.detach().cpu()[:, 0], x_d.detach().cpu()[:, 1], s=1)
                plt.savefig("1d_d.png")

                plt.clf()
                plt.xlim([-8, 8])
                plt.ylim([-8, 8])
                plt.scatter(x_g.detach().cpu()[:, 0], x_g.detach().cpu()[:, 1], s=1)
                plt.savefig("1d_g.png")

                plt.clf()
                plt.xlim([-8, 8])
                plt.ylim([-8, 8])
                plt.scatter(dist_samples.detach().cpu().squeeze()[:, 0],
                            dist_samples.detach().cpu().squeeze()[:, 1],
                            s=1)
                plt.savefig("1d_dist.png")

    if args.estimator == "hmc":
        burn_in = int(floor(args.num_samples_posterior / 2))
        num_samples_posterior = int(ceil(args.num_samples_posterior / 2))
        # burn_in = args.num_samples_posterior
        # num_samples_posterior = args.num_samples_posterior
    else:
        burn_in = 0
        num_samples_posterior = args.num_samples_posterior

    # train the estimator
    utils.print_log(f"Training {args.estimator} sampler", args)
    if args.estimator == "hmc":
        for i in range(500):
            x_g, h_g = generator.sample(args.batch_size)
            *_, accept = generator.entropy_obj(x_g, h_g,
                                               burn_in=burn_in,
                                               num_samples_posterior=num_samples_posterior,
                                               return_accept=True)
            utils.print_log(f"{i} | "
                            f"Accept rate "
                            f"[0]: {accept[0].item():.2f} | "
                            f"mean: {accept.mean().item():.2f} | "
                            f"step size {generator.stepsize.item():.2e}", args)

        # freeze the HMC sampler params (the inf sampler can be frozen with the learn_post_sigma argument)
        generator.mcmc_lr = 0
    else:
        for i in range(500):
            x_g, h_g = generator.sample(args.batch_size)
            generator.entropy_obj(x_g, h_g,
                                  learn_post_sigma=True)
            utils.print_log(f"{i} | "
                            f"Posterior logsigma "
                            f"[0]: {generator.post_logsigma[0].item():.2f} | "
                            f"mean: {generator.post_logsigma.mean().item():.2f}", args)

    # run estimator on the CPU
    generator.to(cpu_device)

    # generate samples (exact!)
    x_g, h_g = generator.sample(args.est_batch_size)

    x_g = x_g.detach()
    h_g = h_g.detach()

    # compute the true value
    x_g_grad = x_g.requires_grad_()
    dist = lgen_inf.dist(cpu_device)
    true_score, = torch.autograd.grad(dist.log_prob(x_g_grad).sum(), x_g_grad)

    # tile x_g and h_g to get multiple samples to estimate the bias
    x_g_tiled = x_g.repeat(1, args.num_samples).view(-1, x_g.size(-1))
    h_g_tiled = h_g.repeat(1, args.num_samples).view(-1, h_g.size(-1))

    # get the inf estimate (and don't update the post_logsigma parameter)
    if args.estimator == "hmc":
        *_, estimator_score = generator.entropy_obj(x_g_tiled, h_g_tiled,
                                                    return_score=True,
                                                    burn_in=burn_in,
                                                    num_samples_posterior=num_samples_posterior)
    else:
        *_, estimator_score = generator.entropy_obj(x_g_tiled, h_g_tiled,
                                                    return_score=True,
                                                    learn_post_sigma=False,
                                                    num_samples_posterior=num_samples_posterior)

    # tile it so we get (batch, samples, features) on its own axis
    estimator_score = torch.cat(estimator_score[None].split(args.num_samples, dim=1), dim=0)

    # take average to get the estimate
    estimator_score_mean = estimator_score.mean(1)
    estimator_score_var = estimator_score.var(1)

    # flip the sign!!!
    estimator_score_mean = -estimator_score_mean

    bias = (true_score - estimator_score_mean).mean()
    var = estimator_score_var.mean()

    utils.print_log(f"Bias: {bias:.4e} | Var: {var:.4e}", args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluating Bias and Variance of Estimators")

    parser.add_argument("--log_file", type=str, default="log.txt")
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--estimator", type=str, required=True, choices=["inf", "hmc"])
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "toy"])

    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for training PCA and Estimator")
    parser.add_argument("--est_batch_size", type=int, default=200, help="Batch size for evaluating estimator")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to take when estimating bias")

    parser.add_argument("--num_samples_posterior", type=int, default=20, help="Used for Inf ent estimator")

    parser.add_argument("--check_dist_eq", action="store_true", default=False, help="Check that distributions match")
    parser.add_argument("--plot_samples", action="store_true", default=False, help="Plot samples from model")

    parser.add_argument("--load_pca", action="store_true", default=False,
                        help="Load PCA from pickle instead of recomputing")
    parser.add_argument("--load_pca_params", action="store_true", default=False,
                        help="Load PCA params from pickle instead of recomputing")

    parser.add_argument("--ckpt_path")  # just for compatibility

    args = parser.parse_args()

    utils.makedirs(args.save_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    utils.print_log('Using {} GPU(s).'.format(torch.cuda.device_count()), args)

    main(args)
