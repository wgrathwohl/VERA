# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import models.wideresnet as wideresnet
import utils
from models.get_models import get_models

# Sampling
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3
n_classes = 10


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, 10)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z)


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None):
        super(CCF, self).__init__(depth, width, norm=norm)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return torch.gather(logits, 1, y[:, None])


def cycle(loader):
    while True:
        for data in loader:
            yield data


def init_random(bs):
    return torch.FloatTensor(bs, 3, 32, 32).uniform_(-1, 1)


def sample_p_0(device, replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random(bs), []
    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // n_classes
    inds = torch.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
        assert not args.uncond, "Can't drawn conditional samples without giving me y"
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs)
    choose_random = (torch.rand(bs) < args.reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(args, device, f, replay_buffer, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    bs = args.batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(device, replay_buffer, bs=bs, y=y)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(args.n_steps):
        f_prime = torch.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
        x_k.data += args.sgld_lr * f_prime + args.sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples

def refine_MALA(logp_net, g, x_g, sgld_lr_z):
    # latent space sgld
    eps_sgld = torch.randn_like(x_g)
    z_sgld = torch.randn((eps_sgld.size(0), args.noise_dim)).to(eps_sgld.device)
    vs = (z_sgld.requires_grad_(), eps_sgld.requires_grad_())
    steps = [vs]
    accepts = []
    gfn = lambda z, e: g.g(z) + g.logsigma.exp() * e
    efn = lambda z, e: logp_net(gfn(z, e)).squeeze()
    with torch.no_grad():
        x_init = gfn(z_sgld, eps_sgld)
    # plot(("{}/{:0%d}_init.png" % niters_digs).format(gen_sgld_dir, itr),
    #      x_init.view(x_g.size(0), *args.data_size))
    for k in range(args.n_sample_steps):
        vs, a = utils.hmc.MALA(vs, efn, sgld_lr_z)
        steps.append(vs)
        accepts.append(a.item())
        print('...', k)
    ar = np.mean(accepts)
    utils.print_log("latent eps accept rate: {}".format(ar), args)
    sgld_lr_z = sgld_lr_z + args.mcmc_lr * (ar - .57) * sgld_lr_z
    z_sgld, eps_sgld = steps[-1]
    with torch.no_grad():
        x_ref = gfn(z_sgld, eps_sgld)
    # plot(("{}/{:0%d}_ref.png" % niters_digs).format(gen_sgld_dir, itr),
    #      x_sgld.view(x_g.size(0), *args.data_size))

    return x_init, x_ref, sgld_lr_z


def uncond_samples(f, g, args, device, save=True):
    _, _, plot = utils.get_data(args)
    sgld_lr = args.sgld_lr
    cond_samples_init = [[] for _ in range(args.n_classes)]
    cond_samples_ref = [[] for _ in range(args.n_classes)]
    for i in range(args.n_sample_batches):
        x_g, h_g = g.sample(args.batch_size, requires_grad=True)
        x_init, x_ref, sgld_lr = refine_MALA(f, g, x_g, sgld_lr)
        x_init = x_init.detach()
        x_ref = x_ref.detach()
        print(sgld_lr)
        plot('{}/samples_{}_init.png'.format(args.save_dir, i), x_init)
        plot('{}/samples_{}_ref.png'.format(args.save_dir, i), x_ref)
        print(i)

        with torch.no_grad():
            _, logits_init = f(x_init, return_logits=True)
            _, logits_ref = f(x_ref, return_logits=True)

            for j in range(logits_init.size(0)):
                y = torch.argmax(logits_init[j])
                cond_samples_init[y].append(x_init[j][None].detach().cpu())

            for j in range(logits_ref.size(0)):
                y = torch.argmax(logits_ref[j])
                cond_samples_ref[y].append(x_ref[j][None].detach().cpu())

        for y in range(args.n_classes):
            x = cond_samples_init[y]
            if len(x) > 0:
                x = torch.cat(x)
                plot('{}/samples_init_class_{}.png'.format(args.save_dir, y), x)
        for y in range(args.n_classes):
            x = cond_samples_ref[y]
            if len(x) > 0:
                x = torch.cat(x)
                plot('{}/samples_ref_class_{}.png'.format(args.save_dir, y), x)

    init_classes = [torch.cat(cls) for cls in cond_samples_init]
    with open("{}/cond_images_init.pkl".format(args.save_dir), 'wb') as f:
        pickle.dump(init_classes, f)

    ref_classes = [torch.cat(cls) for cls in cond_samples_ref]
    with open("{}/cond_images_ref.pkl".format(args.save_dir), 'wb') as f:
        pickle.dump(ref_classes, f)


def cond_samples(f, replay_buffer, args, device):
    _, _, plot = utils.get_data(args)
    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    n_it = replay_buffer.size(0) // 100
    all_y = []
    for i in range(n_it):
        x = replay_buffer[i * 100: (i + 1) * 100].to(device)
        y = f.classify(x).max(1)[1]
        all_y.append(y)

    all_y = torch.cat(all_y, 0)
    each_class = [replay_buffer[all_y == l] for l in range(10)]
    print([len(c) for c in each_class])
    for i in range(100):
        this_im = []
        for l in range(10):
            this_l = each_class[l][i * 10: (i + 1) * 10]
            this_im.append(this_l)
        this_im = torch.cat(this_im, 0)
        if this_im.size(0) > 0:
            plot('{}/samples_{}.png'.format(args.save_dir, i), this_im)
        print(i)


def logp_hist(f, args, device):
    sns.set()
    plt.switch_backend('agg')

    def sample(x, n_steps=args.n_steps):
        x_k = torch.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * torch.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    def grad_norm(x):
        x_k = torch.autograd.Variable(x, requires_grad=True)
        f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    def score_fn(x):
        if args.score_fn == "px":
            return f(x).detach().cpu()
        elif args.score_fn == "py":
            return nn.Softmax()(f.classify(x)).max(1)[0].detach().cpu()
        elif args.score_fn == "pxgrad":
            return -torch.log(grad_norm(x).detach().cpu())
        elif args.score_fn == "refine":
            init_score = f(x)
            x_r = sample(x)
            final_score = f(x_r)
            delta = init_score - final_score
            return delta.detach().cpu()
        elif args.score_fn == "refinegrad":
            init_score = -grad_norm(x).detach()
            x_r = sample(x)
            final_score = -grad_norm(x_r).detach()
            delta = init_score - final_score
            return delta.detach().cpu()
        elif args.score_fn == "refinel2":
            x_r = sample(x)
            norm = (x - x_r).view(x.size(0), -1).norm(p=2, dim=1)
            return -norm.detach().cpu()
        else:
            return f.classify(x).max(1)[0].detach().cpu()
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * torch.randn_like(x)]
    )
    datasets = {
        "cifar10": tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False),
        "svhn": tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test"),
        "cifar100": tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=False),
        "celeba": tv.datasets.CelebA(root="./data",
                                     split="test",
                                     transform=tr.Compose([tr.Resize(32),
                                                           tr.ToTensor(),
                                                           tr.Normalize((.5, .5, .5), (.5, .5, .5)),
                                                           lambda x: x + args.sigma * torch.randn_like(x)]),
                                     download=False)
    }

    score_dict = {}
    for dataset_name in args.datasets:
        print(dataset_name)
        dataset = datasets[dataset_name]
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4, drop_last=False)
        this_scores = []
        for x, _ in dataloader:
            x = x.to(device)
            scores = score_fn(x)
            print(scores.mean())
            this_scores.extend(scores.numpy())
        score_dict[dataset_name] = this_scores

    for name, scores in score_dict.items():
        plt.hist(scores, label=name, bins=100, normed=True, alpha=.5)
    plt.legend()
    plt.savefig(args.save_dir + "/fig.pdf")


def OODAUC(f, args, device):
    print("OOD Evaluation")

    def grad_norm(x):
        x_k = torch.autograd.Variable(x, requires_grad=True)
        f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * torch.randn_like(x)]
    )

    dset_real = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)
    dload_real = DataLoader(dset_real, batch_size=100, shuffle=False, num_workers=4, drop_last=False)

    if args.ood_dataset == "svhn":
        dset_fake = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test")
    elif args.ood_dataset == "cifar_100":
        dset_fake = tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=False)
    elif args.ood_dataset == "celeba":
        dset_fake = tv.datasets.CelebA(root="./data", split="test",
                                       transform=tr.Compose([tr.Resize(32),
                                                             tr.ToTensor(),
                                                             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
                                                             lambda x: x + args.sigma * torch.randn_like(x)]),
                                       download=False)
    else:
        dset_fake = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)

    dload_fake = DataLoader(dset_fake, batch_size=100, shuffle=True, num_workers=4, drop_last=False)
    print(len(dload_real), len(dload_fake))
    real_scores = []
    print("Real scores...")

    def score_fn(x):
        if args.score_fn == "px":
            return f(x).detach().cpu()
        elif args.score_fn == "py":
            return nn.Softmax()(f.classify(x)).max(1)[0].detach().cpu()
        else:
            return -grad_norm(x).detach().cpu()

    for x, _ in dload_real:
        x = x.to(device)
        scores = score_fn(x)
        real_scores.append(scores.numpy())
        print(scores.mean())
    fake_scores = []
    print("Fake scores...")
    if args.ood_dataset == "cifar_interp":
        last_batch = None
        for i, (x, _) in enumerate(dload_fake):
            x = x.to(device)
            if i > 0:
                x_mix = (x + last_batch) / 2 + args.sigma * torch.randn_like(x)
                scores = score_fn(x_mix)
                fake_scores.append(scores.numpy())
                print(scores.mean())
            last_batch = x
    elif args.ood_dataset == "uniform":
        for i, (x, _) in enumerate(dload_real):
            x = x.to(device)
            x = torch.rand_like(x) * 2. - 1.
            scores = score_fn(x)
            fake_scores.append(scores.numpy())
            print(scores.mean())
    elif args.ood_dataset == "constant":
        for i, (x, _) in enumerate(dload_real):
            x = x.to(device)
            x = torch.zeros_like(x) * 2. - 1.
            scores = score_fn(x)
            fake_scores.append(scores.numpy())
            print(scores.mean())
    else:
        for i, (x, _) in enumerate(dload_fake):
            x = x.to(device)
            scores = score_fn(x)
            fake_scores.append(scores.numpy())
            print(scores.mean())
    real_scores = np.concatenate(real_scores)
    fake_scores = np.concatenate(fake_scores)
    real_labels = np.ones_like(real_scores)
    fake_labels = np.zeros_like(fake_scores)
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([real_labels, fake_labels])
    score = sklearn.metrics.roc_auc_score(labels, scores)
    print(score)


def test_clf(f, args, device):

    def sample(x, n_steps=args.n_steps):
        x_k = torch.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * torch.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    if args.test_dataset == "cifar_train":
        args.dataset = "cifar10"
        dload, _, _ = utils.get_data(args)
    elif args.test_dataset == "cifar_test":
        args.dataset = "cifar10"
        _, dload, _ = utils.get_data(args)
    else:
        raise ValueError

    f.eval()
    corrects, losses, pys, preds = [], [], [], []
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        if args.n_steps > 0:
            x_p_d = sample(x_p_d)
        _, logits = f(x_p_d, return_logits=True)
        py = nn.Softmax()(f.classify(x_p_d)).max(1)[0].detach().cpu().numpy()
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().detach().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        pys.extend(py)
        preds.extend(logits.max(1)[1].cpu().numpy())

    loss = np.mean(losses)
    correct = np.mean(corrects)
    torch.save({"losses": losses, "corrects": corrects, "pys": pys}, os.path.join(args.save_dir, "vals.pt"))
    print(loss, correct)


def main(args):
    utils.makedirs(args.save_dir)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    f, g = get_models(args)

    print(f"loading model from {args.ckpt_path}")

    # load em up
    ckpt = torch.load(args.ckpt_path)
    f.load_state_dict(ckpt["model"]["logp_net"])
    g.load_state_dict(ckpt["model"]["g"])

    f = f.to(device)
    g = g.to(device)
    f.eval()

    if args.eval == "OOD":
        OODAUC(f, args, device)
    elif args.eval == "test_clf":
        test_clf(f, args, device)
    elif args.eval == "cond_samples":
        cond_samples(f, g, args, device)
    elif args.eval == "uncond_samples":
        uncond_samples(f, g, args, device)
    elif args.eval == "logp_hist":
        logp_hist(f, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluating Samples from EBM")
    parser.add_argument("--eval", default="OOD", type=str,
                        choices=["uncond_samples", "cond_samples", "logp_hist", "OOD", "test_clf"])
    parser.add_argument("--score_fn", default="px", type=str,
                        choices=["px", "py", "pxgrad"], help="For OODAUC, cho¸oses what score function we use.")
    parser.add_argument("--ood_dataset", default="svhn", type=str,
                        choices=["svhn", "cifar_interp", "cifar_100", "celeba", "uniform", "constant"],
                        help="Chooses which dataset to compare against for OOD")
    parser.add_argument("--dataset", default="cifar_test", type=str,
                        help="Dataset to use when running test_clf for classification accuracy")
    parser.add_argument("--test_dataset", default="cifar_test", type=str,
                        choices=["cifar_train", "cifar_test", "svhn_test", "svhn_train"],
                        help="Dataset to use when running test_clf for classification accuracy")
    parser.add_argument("--datasets", nargs="+", type=str, default=[],
                        help="The datasets you wanna use to generate a log p(x) histogram")
    # optimization
    parser.add_argument("--batch_size", type=int, default=64)
    # regularization
    parser.add_argument("--sigma", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"])
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=0)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    parser.add_argument("--mcmc_lr", type=float, default=.02)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='YOUR_SAVE_PATH_BUDDDDDDYYYYYYY')
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_sample_steps", type=int, default=100)
    parser.add_argument("--n_sample_batches", type=int, default=100)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--generator_type", type=str, default="vlvm", choices=["lvm", "vlvm"])
    parser.add_argument("--noise_dim", type=int, default=128)
    parser.add_argument("--unit_interval", action="store_true")
    parser.add_argument("--data_aug", action="store_true")

    parser.add_argument("--g_feats", type=int, default=128)
    parser.add_argument("--post_lr", type=float, default=.02)
    parser.add_argument("--log_file", type=str, default="log.txt")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--logit", action="store_true")

    args = parser.parse_args()

    args.clf = True

    if args.depth == 28 and args.width == 10:
        args.thicc_resnet = True
        args.wide_resnet = False
        args.resnet = False
    elif args.depth == 28 and args.width == 2:
        args.wide_resnet = True
        args.thicc_resnet = False
        args.resnet = False
    else:
        args.wide_resnet = False
        args.thicc_resnet = False
        args.resnet = True

    if args.dataset == "cifar10":
        args.dropout = .3
    else:
        args.dropout = .4

    if args.dataset == "cifar100":
        args.n_classes = 100
    else:
        args.n_classes = 10

    main(args)
