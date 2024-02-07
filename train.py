"""
EBM training.
"""

import argparse
import json
import os
import time
from operator import itemgetter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import numpy as np
import torch.utils
from torch.utils.data import DataLoader
from torch import distributions
import torchvision
import torch.nn.functional as F
import utils
from models.get_models import get_models
from models.jem import get_buffer, sample_q
from utils.toy_data import TOY_DSETS
from tabular import TAB_DSETS

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def brier_score_loss_multi(y_true, y_prob):
    """
    Brier score for multiclass.
    https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    """
    return ((y_prob - y_true) ** 2).sum(1).mean()


def main(args):
    """
    Main function.
    """
    data_sgld_dir, gen_sgld_dir, z_sgld_dir, \
    data_sgld_chain_dir, gen_sgld_chain_dir, z_sgld_chain_dir, \
    save_model_dir = utils.make_logdirs(args)

    logp_net, g = get_models(args)
    replay_buffer = get_buffer(args)

    g.train()
    g.to(device)
    logp_net.train()
    logp_net.to(device)

    # data
    train_loader, test_loader, plot = utils.get_data(args)

    batches_per_epoch = len(train_loader)
    niters = batches_per_epoch * args.n_epochs
    niters_digs = np.ceil(np.log10(niters)) + 1

    if args.ssl:
        labeled_dataset = utils.ssl.labeled_subset(train_loader.dataset,
                                                   args.labels_per_class,
                                                   args.seed,
                                                   args.num_classes)
        labeled_loader = DataLoader(labeled_dataset, min(args.batch_size, len(labeled_dataset)),
                                    shuffle=True, drop_last=True)
        labeled_loader = utils.ssl.cycle(labeled_loader)

    # optimization
    e_optimizer = torch.optim.Adam(logp_net.parameters(),
                                   lr=args.lr, betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    g_optimizer = torch.optim.Adam(list(g.parameters()),
                                   lr=args.glr, betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    scheduler_kwargs = {
        "milestones": [int(epoch * batches_per_epoch) for epoch in args.decay_epochs],
        "gamma": args.decay_rate
    }
    e_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(e_optimizer, **scheduler_kwargs)
    g_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, **scheduler_kwargs)

    itr = 0
    start_epoch = 0
    train_accs = []
    test_accs = []
    aucs = []
    briers = []
    test_lps = []
    modes = []
    kl = []
    eval_itrs = []
    try:
        ckpt = torch.load(args.ckpt_path)
        start_epoch = ckpt["epoch"]
        train_accs = ckpt["train_accs"]
        test_accs = ckpt["test_accs"]
        aucs = ckpt["aucs"]
        briers = ckpt["briers"]
        if "test_lps" in ckpt:
            test_lps = ckpt["test_lps"]
        else:
            test_lps = []
        modes = ckpt["modes"]
        kl = ckpt["kl"]
        eval_itrs = ckpt["eval_itrs"]
        itr = ckpt["itr"]
        logp_net.load_state_dict(ckpt["model"]["logp_net"])
        g.load_state_dict(ckpt["model"]["g"])
        e_optimizer.load_state_dict(ckpt["optimizer"]["e"])
        g_optimizer.load_state_dict(ckpt["optimizer"]["g"])
        e_lr_scheduler.load_state_dict(ckpt["scheduler"]["e"])
        g_lr_scheduler.load_state_dict(ckpt["scheduler"]["g"])
    except IOError:
        utils.print_log("no checkpoint given", args)

    def save_ckpt(itr, overwrite=True, prefix=""):
        """
        Save a checkpoint in case job is prempted.
        """
        if overwrite and prefix == "":
            # overwrite the same checkpoint since it's just used for preemption
            path = args.ckpt_path
        elif overwrite and prefix != "":
            path = os.path.join(save_model_dir, "{}.pt".format(prefix))
        else:
            path = os.path.join(save_model_dir, "{}_{:06d}.pt".format(prefix, itr))
        # ckpt_path will be made automatically on v2
        try:
            logp_net.cpu()
            g.cpu()
            torch.save({
                # if last batch in epoch, go to next one
                "epoch": epoch + 1 if itr % batches_per_epoch == 0 else epoch,
                "train_accs": train_accs,
                "test_accs": test_accs,
                "aucs": aucs,
                "briers": briers,
                "test_lps": test_lps,
                "modes": modes,
                "kl": kl,
                "eval_itrs": eval_itrs,
                "itr": itr,
                "model": {
                    "logp_net": logp_net.state_dict(),
                    "g": g.state_dict()
                },
                "optimizer": {
                    "e": e_optimizer.state_dict(),
                    "g": g_optimizer.state_dict()
                },
                "scheduler": {
                    "e": e_lr_scheduler.state_dict(),
                    "g": g_lr_scheduler.state_dict()
                }
            }, path)
            logp_net.to(device)
            g.to(device)
        except IOError:
            utils.print_log("Unable to save %s %d" % (path, itr), args)

    sgld_lr = 1. / args.noise_dim
    sgld_lr_z = 1. / args.noise_dim
    sgld_lr_zne = 1. / args.noise_dim

    entropy_obj = torch.tensor(0.)
    grad_ld = torch.tensor(0.)
    logq_obj = torch.tensor(0.)
    logp_obj = torch.tensor(0.)
    ld = torch.tensor(0.)
    lg_detach = torch.tensor(0.)
    ebm_gn, ent_gn = torch.tensor(0.), torch.tensor(0.)

    c_loss, train_acc, auc, brier, unsup_ent = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), \
                                                     torch.tensor(0.), torch.tensor(0.)

    t = time.time()

    for epoch in range(start_epoch, args.n_epochs):

        for x_d, y_d in train_loader:
            if args.dataset in TOY_DSETS:
                x_d = utils.toy_data.inf_train_gen(args.dataset, batch_size=args.batch_size)
                x_d = torch.from_numpy(x_d).float().to(device)
            else:
                x_d = x_d.to(device)

            if args.ssl:
                # ssl
                x_l, y_l = labeled_loader.__next__()
            else:
                # full labels
                x_l, y_l = x_d, y_d

            x_l = x_l.to(device)
            x_l.requires_grad_()
            x_d.requires_grad_()
            y_l = y_l.to(device)

            # warmup lr
            if itr < args.warmup_iters:
                lr = args.lr * (itr + 1) / float(args.warmup_iters)
                glr = args.glr * (itr + 1) / float(args.warmup_iters)
                for param_group in e_optimizer.param_groups:
                    param_group['lr'] = lr
                for param_group in g_optimizer.param_groups:
                    param_group['lr'] = glr

            if args.clf_only:
                ld, ld_logits = logp_net(x_l, return_logits=True)

                c_loss = torch.nn.CrossEntropyLoss()(ld_logits, y_l)

                # calculate accuracy
                chosen = ld_logits.max(1).indices
                train_acc = (chosen == y_l).float().mean().item()

                # calculate AUC and brier
                class_probs = torch.nn.functional.softmax(ld_logits.detach(), dim=1)
                if args.num_classes == 2:
                    auc = roc_auc_score(y_true=y_l.cpu(), y_score=class_probs[:, 1].cpu())
                    brier = brier_score_loss(y_true=y_l.cpu(), y_prob=class_probs[:, 1].cpu())
                else:
                    targets = torch.zeros((y_l.size(0), args.num_classes)).to(device)
                    targets.scatter_(1, y_l[:, None], 1)
                    brier = brier_score_loss_multi(y_true=targets, y_prob=class_probs).cpu()

                if args.pg_control > 0:
                    grad_ld = torch.autograd.grad(ld.sum(), x_d,
                                                  create_graph=True)[0].flatten(start_dim=1).norm(2, 1)

                e_optimizer.zero_grad()
                (args.clf_weight * c_loss + (grad_ld ** 2. / 2.).mean() * args.pg_control).backward()
                e_optimizer.step()

                # decay learning rate
                e_lr_scheduler.step()

                itr += 1

                if itr % args.print_every == 0:

                    # get info to log
                    new_time = time.time()
                    elapsed = new_time - t
                    t = new_time

                    curr_e_lr, = e_lr_scheduler.get_last_lr()

                    utils.print_log("{:.2e} / iter ({}) | "
                                    "clf obj: {:.4e} ({:.4f}) ({:.4f}) ({:.4f}) | "
                                    "e-lr {:.2e}".format(
                        elapsed / args.print_every, itr,
                        c_loss.item(), train_acc, auc, brier,
                        curr_e_lr), args)

            elif args.maximum_likelihood or args.ssm:
                if args.maximum_likelihood:
                    loss = -logp_net(x_d).mean()
                else:
                    x_d.requires_grad_()
                    lpx = logp_net(x_d)
                    dldx = torch.autograd.grad(lpx.sum(), x_d, create_graph=True)[0]
                    eps = torch.randn_like(x_d)
                    eH = torch.autograd.grad(dldx, x_d, grad_outputs=eps, create_graph=True)[0]
                    eHe = eH * eps
                    trH = eHe.sum(1)
                    loss = (trH + (dldx * dldx).sum(1) * .5).mean()

                e_optimizer.zero_grad()
                loss.backward()
                e_optimizer.step()

                if itr % args.print_every == 0:
                    utils.print_log("iter ({}) | "
                                    "logp(x): {:.4e}".format(
                        itr, -loss.item()), args)

                if itr % args.viz_every == 0:
                    if args.dataset in TOY_DSETS:
                        " DO plotting of my posterior and the true posterior!!! "
                        plt.clf()
                        xg = logp_net.sample(args.batch_size).detach().cpu().numpy()
                        xd = x_d.detach().cpu().numpy()

                        ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
                        ax.scatter(xg[:, 0], xg[:, 1], s=1)

                        ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
                        ax.scatter(xd[:, 0], xd[:, 1], s=1)

                        logp_net.cpu()

                        ax = plt.subplot(1, 4, 3, aspect="equal")
                        utils.viz.plt_toy_density(lambda x: logp_net(x), ax,
                                                  low=-4, high=4,
                                                  title="p(x)")

                        ax = plt.subplot(1, 4, 4, aspect="equal")
                        utils.viz.plt_toy_density(lambda x: logp_net(x), ax,
                                                  low=-4, high=4,
                                                  exp=False, title="log p(x)")

                        plt.savefig(("{}/{:0%d}.png" % niters_digs).format(data_sgld_dir, itr))

                        logp_net.to(device)

                    elif args.dataset in TAB_DSETS:
                        pass

                    else:
                        x_mog = logp_net.sample(args.batch_size)
                        plot(("{}/{:0%d}_MOG.png" % niters_digs).format(data_sgld_dir, itr),
                             x_mog.view(x_mog.size(0), *args.data_size))
                itr += 1

            elif args.vat:
                _, ld_logits = logp_net(x_l, return_logits=True)
                c_loss = torch.nn.CrossEntropyLoss()(ld_logits, y_l)

                _, unsup_logits = logp_net(x_d, return_logits=True)
                unsup_ent = distributions.Categorical(logits=unsup_logits).entropy().mean()

                vat_loss = utils.VATLoss(xi=10.0, eps=args.vat_eps, ip=1)
                lds = vat_loss(logp_net, x_d)

                e_optimizer.zero_grad()
                (args.clf_weight * c_loss + args.clf_ent_weight * unsup_ent + args.vat_weight * lds).backward()
                e_optimizer.step()

                # decay learning rates
                e_lr_scheduler.step()
                g_lr_scheduler.step()

                itr += 1

                if itr % args.print_every == 0:

                    # get some info to log
                    new_time = time.time()
                    elapsed = new_time - t
                    t = new_time

                    if args.generator_type == "vera":
                        stepsize = g.stepsize
                        post_sigma = 0
                    else:
                        stepsize = 0
                        post_sigma = g.post_logsigma.exp().mean().item()

                    curr_e_lr, = e_lr_scheduler.get_last_lr()
                    curr_g_lr, = g_lr_scheduler.get_last_lr()

                    utils.print_log("{:.1e} s/itr ({}) | "
                                    "clf obj: {:.2e} ({:.4f}) ({:.4f}) ({:.4f}) | "
                                    "log p obj = {:.2e}, log q obj = {:.2e}, sigma = {:.2e} | "
                                    "log p(x_d) = {:.2e}, log p(x_m) = {:.2e}, ent = {:.2e} | "
                                    "sgld_lr = {:.1e}, sgld_lr_z = {:.1e}, sgld_lr_zne = {:.1e} | "
                                    "stepsize = {:.1e}, post_sigma = {:.1e} | "
                                    "ebm gn = {:.1e}, ent gn = {:.1e} | "
                                    "e-lr {:.2e}, g-lr {:.2e}".format(
                            elapsed / args.print_every, itr,
                            c_loss.item(), train_acc, auc, brier,
                            logp_obj.item(), logq_obj.item(), g.logsigma.exp().item(),
                            ld.mean().item(), lg_detach.mean().item(), entropy_obj,
                            sgld_lr, sgld_lr_z, sgld_lr_zne,
                            stepsize, post_sigma,
                            ebm_gn.item(), ent_gn.item(),
                            curr_e_lr, curr_g_lr), args)

            elif args.jem_baseline:
                if args.ssl:
                    ld, unsup_logits = logp_net(x_d, return_logits=True)
                    _, ld_logits = logp_net(x_l, return_logits=True)
                    unsup_ent = distributions.Categorical(logits=unsup_logits).entropy()
                elif args.jem:
                    ld, ld_logits = logp_net(x_d, return_logits=True)
                else:
                    ld, ld_logits = logp_net(x_d).squeeze(), torch.tensor(0.).to(device)

                grad_ld = torch.autograd.grad(ld.sum(), x_d,
                                              create_graph=True)[0].flatten(start_dim=1).norm(2, 1)

                x_g = sample_q(logp_net, replay_buffer,
                               args.batch_size, args.n_steps, args.sgld_lr, args.sgld_std, args.reinit_freq, device)
                lg_detach = logp_net(x_g).squeeze()

                logp_obj = (ld - lg_detach).mean()
                e_loss = -logp_obj + \
                         (ld ** 2).mean() * args.p_control + \
                         (lg_detach ** 2).mean() * args.n_control + \
                         (grad_ld ** 2. / 2.).mean() * args.pg_control + \
                         unsup_ent.mean() * args.clf_ent_weight

                if args.clf:
                    c_loss = torch.nn.CrossEntropyLoss()(ld_logits, y_l)

                    chosen = ld_logits.max(1).indices
                    train_acc = (chosen == y_l).float().mean().item()

                    class_probs = torch.nn.functional.softmax(ld_logits.detach(), dim=1)
                    if args.num_classes == 2:
                        auc = roc_auc_score(y_true=y_l.cpu(), y_score=class_probs[:, 1].cpu())
                        brier = brier_score_loss(y_true=y_l.cpu(), y_prob=class_probs[:, 1].cpu())
                    else:
                        targets = torch.zeros((y_l.size(0), args.num_classes)).to(device)
                        targets.scatter_(1, y_l[:, None], 1)
                        brier = brier_score_loss_multi(y_true=targets, y_prob=class_probs).cpu()

                e_optimizer.zero_grad()
                (e_loss + args.clf_weight * c_loss).backward()
                e_optimizer.step()

                # decay learning rates
                e_lr_scheduler.step()

                itr += 1

                if itr % args.print_every == 0:

                    # get some info to log
                    new_time = time.time()
                    elapsed = new_time - t
                    t = new_time

                    curr_e_lr, = e_lr_scheduler.get_last_lr()

                    utils.print_log("{:.1e} s/itr ({}) | "
                                    "clf obj: {:.2e} ({:.4f}) ({:.4f}) ({:.4f}) | "
                                    "log p obj = {:.2e}, log q obj = {:.2e}, sigma = {:.2e} | "
                                    "log p(x_d) = {:.2e}, log p(x_m) = {:.2e}, ent = {:.2e} | "
                                    "sgld_lr = {:.1e}, sgld_lr_z = {:.1e}, sgld_lr_zne = {:.1e} | "
                                    "ebm gn = {:.1e}, ent gn = {:.1e} | "
                                    "e-lr {:.2e}".format(
                        elapsed / args.print_every, itr,
                        c_loss.item(), train_acc, auc, brier,
                        logp_obj.item(), logq_obj.item(), g.logsigma.exp().item(),
                        ld.mean().item(), lg_detach.mean().item(), entropy_obj,
                        sgld_lr, sgld_lr_z, sgld_lr_zne,
                        ebm_gn.item(), ent_gn.item(),
                        curr_e_lr), args)

                if itr % args.viz_every == 0:
                    if args.dataset in TOY_DSETS:
                        " DO plotting of my posterior and the true posterior!!! "
                        plt.clf()
                        xg = x_g.detach().cpu().numpy()
                        xd = x_d.detach().cpu().numpy()

                        ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
                        ax.scatter(xg[:, 0], xg[:, 1], s=1)

                        ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
                        ax.scatter(xd[:, 0], xd[:, 1], s=1)

                        logp_net.cpu()

                        ax = plt.subplot(1, 4, 3, aspect="equal")
                        utils.viz.plt_toy_density(lambda x: logp_net(x), ax,
                                                  low=-4, high=4,
                                                  title="p(x)")

                        ax = plt.subplot(1, 4, 4, aspect="equal")
                        utils.viz.plt_toy_density(lambda x: logp_net(x), ax,
                                                  low=-4, high=4,
                                                  exp=False, title="log p(x)")

                        plt.savefig(("{}/{:0%d}.png" % niters_digs).format(data_sgld_dir, itr))

                        logp_net.to(device)

                    elif args.dataset in TAB_DSETS:
                        pass

                    else:
                        plot(("{}/{:0%d}.png" % niters_digs).format(data_sgld_dir, itr),
                             x_g.view(x_g.size(0), *args.data_size))

                        if args.mog_comps is not None or args.nice:
                            x_mog = logp_net.sample(args.batch_size)
                            plot(("{}/{:0%d}_MOG.png" % niters_digs).format(data_sgld_dir, itr),
                                 x_mog.view(x_mog.size(0), *args.data_size))

            else:

                # sample from q(x, h)
                x_g, h_g = g.sample(args.batch_size, requires_grad=True)

                # ebm (contrastive divergence) objective
                if itr % args.e_iters == 0:
                    x_g_detach = x_g.detach().requires_grad_()
                    if args.no_g_batch_norm:
                        logp_net.apply(utils.set_bn_to_eval)
                    lg_detach = logp_net(x_g_detach).squeeze()
                    if args.no_g_batch_norm:
                        logp_net.apply(utils.set_bn_to_train)
                    if args.ssl:
                        ld, unsup_logits = logp_net(x_d, return_logits=True)
                        _, ld_logits = logp_net(x_l, return_logits=True)
                        unsup_ent = distributions.Categorical(logits=unsup_logits).entropy()
                    elif args.jem:
                        ld, ld_logits = logp_net(x_d, return_logits=True)
                    else:
                        ld, ld_logits = logp_net(x_d).squeeze(), torch.tensor(0.).to(device)

                    grad_ld = torch.autograd.grad(ld.sum(), x_d,
                                                  create_graph=True)[0].flatten(start_dim=1).norm(2, 1)

                    logp_obj = (ld - lg_detach).mean()
                    e_loss = -logp_obj + \
                             (ld ** 2).mean() * args.p_control + \
                             (lg_detach ** 2).mean() * args.n_control + \
                             (grad_ld ** 2. / 2.).mean() * args.pg_control + \
                             unsup_ent.mean() * args.clf_ent_weight

                    if args.clf:
                        c_loss = torch.nn.CrossEntropyLoss()(ld_logits, y_l)

                        chosen = ld_logits.max(1).indices
                        train_acc = (chosen == y_l).float().mean().item()

                        class_probs = torch.nn.functional.softmax(ld_logits.detach(), dim=1)
                        if args.num_classes == 2:
                            auc = roc_auc_score(y_true=y_l.cpu(), y_score=class_probs[:, 1].cpu())
                            brier = brier_score_loss(y_true=y_l.cpu(), y_prob=class_probs[:, 1].cpu())
                        else:
                            targets = torch.zeros((y_l.size(0), args.num_classes)).to(device)
                            targets.scatter_(1, y_l[:, None], 1)
                            brier = brier_score_loss_multi(y_true=targets, y_prob=class_probs).cpu()

                    e_optimizer.zero_grad()
                    (e_loss + args.clf_weight * c_loss).backward()
                    e_optimizer.step()

                # gen obj
                if itr % args.g_iters == 0:
                    lg = logp_net(x_g).squeeze()
                    grad = torch.autograd.grad(lg.sum(), x_g, retain_graph=True)[0]
                    ebm_gn = grad.norm(2, 1).mean()
                    if args.ent_weight != 0.:
                        entropy_obj, ent_gn = g.entropy_obj(x_g, h_g)

                    logq_obj = lg.mean() + args.ent_weight * entropy_obj

                    g_loss = -logq_obj

                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()

                # clamp sigma to (.01, max_sigma) for generators
                if args.generator_type in ["verahmc", "vera"]:
                    g.clamp_sigma(args.max_sigma, sigma_min=args.min_sigma)

                # decay learning rates
                e_lr_scheduler.step()
                g_lr_scheduler.step()

                itr += 1

                if itr % args.print_every == 0:

                    # get some info to log
                    new_time = time.time()
                    elapsed = new_time - t
                    t = new_time

                    if args.generator_type == "verahmc":
                        stepsize = g.stepsize
                        post_sigma = 0
                    else:
                        stepsize = 0
                        post_sigma = g.post_logsigma.exp().mean().item()

                    curr_e_lr, = e_lr_scheduler.get_last_lr()
                    curr_g_lr, = g_lr_scheduler.get_last_lr()

                    utils.print_log("{:.1e} s/itr ({}) | "
                                    "clf obj: {:.2e} ({:.4f}) ({:.4f}) ({:.4f}) | "
                                    "log p obj = {:.2e}, log q obj = {:.2e}, sigma = {:.2e} | "
                                    "log p(x_d) = {:.2e}, log p(x_m) = {:.2e}, ent = {:.2e} | "
                                    "sgld_lr = {:.1e}, sgld_lr_z = {:.1e}, sgld_lr_zne = {:.1e} | "
                                    "stepsize = {:.1e}, post_sigma = {:.1e} | "
                                    "ebm gn = {:.1e}, ent gn = {:.1e} | "
                                    "e-lr {:.2e}, g-lr {:.2e}".format(
                            elapsed / args.print_every, itr,
                            c_loss.item(), train_acc, auc, brier,
                            logp_obj.item(), logq_obj.item(), g.logsigma.exp().item(),
                            ld.mean().item(), lg_detach.mean().item(), entropy_obj,
                            sgld_lr, sgld_lr_z, sgld_lr_zne,
                            stepsize, post_sigma,
                            ebm_gn.item(), ent_gn.item(),
                            curr_e_lr, curr_g_lr), args)

                if itr % args.viz_every == 0:
                    if args.dataset in TOY_DSETS:
                        " DO plotting of my posterior and the true posterior!!! "
                        plt.clf()
                        xg = x_g.detach().cpu().numpy()
                        xd = x_d.detach().cpu().numpy()

                        ax = plt.subplot(1, 4, 1, aspect="equal", title='gen')
                        ax.scatter(xg[:, 0], xg[:, 1], s=1)

                        ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
                        ax.scatter(xd[:, 0], xd[:, 1], s=1)

                        logp_net.cpu()

                        ax = plt.subplot(1, 4, 3, aspect="equal")
                        utils.viz.plt_toy_density(lambda x: logp_net(x), ax,
                                                  low=-4, high=4,
                                                  title="p(x)")

                        ax = plt.subplot(1, 4, 4, aspect="equal")
                        utils.viz.plt_toy_density(lambda x: logp_net(x), ax,
                                                  low=-4, high=4,
                                                  exp=False, title="log p(x)")

                        plt.savefig(("{}/{:0%d}.png" % niters_digs).format(data_sgld_dir, itr))

                        logp_net.to(device)

                    elif args.dataset in TAB_DSETS:
                        pass

                    else:
                        del x_g, h_g
                        x_g, h_g = g.sample(args.batch_size, requires_grad=True)

                        plot(("{}/{:0%d}_init.png" % niters_digs).format(data_sgld_dir, itr),
                             x_g.view(x_g.size(0), *args.data_size))

                        if args.mog_comps is not None or args.nice:
                            x_mog = logp_net.sample(args.batch_size)
                            plot(("{}/{:0%d}_MOG.png" % niters_digs).format(data_sgld_dir, itr),
                                 x_mog.view(x_mog.size(0), *args.data_size))

                        # input space sgld
                        x_sgld = x_g
                        steps = [x_sgld.detach()]
                        accepts = []
                        for k in range(args.sgld_steps):
                            [x_sgld], a = utils.hmc.MALA([x_sgld], lambda x: logp_net(x).squeeze(), sgld_lr)
                            steps.append(x_sgld.detach())
                            accepts.append(a.item())
                        ar = np.mean(accepts)
                        utils.print_log("data accept rate: {}".format(ar), args)
                        sgld_lr = sgld_lr + args.mcmc_lr * (ar - .57) * sgld_lr
                        plot(("{}/{:0%d}_ref.png" % niters_digs).format(data_sgld_dir, itr),
                             x_sgld.view(x_g.size(0), *args.data_size))

                        chain = torch.cat([step[0][None] for step in steps], 0)
                        plot(("{}/{:0%d}.png" % niters_digs).format(data_sgld_chain_dir, itr),
                             chain.view(chain.size(0), *args.data_size))

                        # latent space sgld
                        eps_sgld = torch.randn_like(x_g)
                        z_sgld = torch.randn((eps_sgld.size(0), args.noise_dim)).to(eps_sgld.device)
                        vs = (z_sgld.requires_grad_(), eps_sgld.requires_grad_())
                        steps = [vs]
                        accepts = []
                        gfn = lambda z, e: g.g(z) + g.logsigma.exp() * e
                        efn = lambda z, e: logp_net(gfn(z, e)).squeeze()
                        with torch.no_grad():
                            x_sgld = gfn(z_sgld, eps_sgld)
                        plot(("{}/{:0%d}_init.png" % niters_digs).format(gen_sgld_dir, itr),
                             x_sgld.view(x_g.size(0), *args.data_size))
                        for k in range(args.sgld_steps):
                            vs, a = utils.hmc.MALA(vs, efn, sgld_lr_z)
                            steps.append(vs)
                            accepts.append(a.item())
                        ar = np.mean(accepts)
                        utils.print_log("latent eps accept rate: {}".format(ar), args)
                        sgld_lr_z = sgld_lr_z + args.mcmc_lr * (ar - .57) * sgld_lr_z
                        z_sgld, eps_sgld = steps[-1]
                        with torch.no_grad():
                            x_sgld = gfn(z_sgld, eps_sgld)
                        plot(("{}/{:0%d}_ref.png" % niters_digs).format(gen_sgld_dir, itr),
                             x_sgld.view(x_g.size(0), *args.data_size))

                        z_steps, eps_steps = zip(*steps)
                        z_chain = torch.cat([step[0][None] for step in z_steps], 0)
                        eps_chain = torch.cat([step[0][None] for step in eps_steps], 0)
                        with torch.no_grad():
                            chain = gfn(z_chain, eps_chain)
                        plot(("{}/{:0%d}.png" % niters_digs).format(gen_sgld_chain_dir, itr),
                             chain.view(chain.size(0), *args.data_size))

                        # latent space sgld no eps
                        z_sgld = torch.randn((eps_sgld.size(0), args.noise_dim)).to(eps_sgld.device)
                        vs = (z_sgld.requires_grad_(),)
                        steps = [vs]
                        accepts = []
                        gfn = lambda z: g.g(z)
                        efn = lambda z: logp_net(gfn(z)).squeeze()
                        with torch.no_grad():
                            x_sgld = gfn(z_sgld)
                        plot(("{}/{:0%d}_init.png" % niters_digs).format(z_sgld_dir, itr),
                             x_sgld.view(x_g.size(0), *args.data_size))
                        for k in range(args.sgld_steps):
                            vs, a = utils.hmc.MALA(vs, efn, sgld_lr_zne)
                            steps.append(vs)
                            accepts.append(a.item())
                        ar = np.mean(accepts)
                        utils.print_log("latent accept rate: {}".format(ar), args)
                        sgld_lr_zne = sgld_lr_zne + args.mcmc_lr * (ar - .57) * sgld_lr_zne
                        z_sgld, = steps[-1]
                        with torch.no_grad():
                            x_sgld = gfn(z_sgld)
                        plot(("{}/{:0%d}_ref.png" % niters_digs).format(z_sgld_dir, itr),
                             x_sgld.view(x_g.size(0), *args.data_size))

                        z_steps = [s[0] for s in steps]
                        z_chain = torch.cat([step[0][None] for step in z_steps], 0)
                        with torch.no_grad():
                            chain = gfn(z_chain)
                        plot(("{}/{:0%d}.png" % niters_digs).format(z_sgld_chain_dir, itr),
                             chain.view(chain.size(0), *args.data_size))

            if itr % args.save_every == 0:
                save_ckpt(itr, overwrite=False)

            # if itr % args.ckpt_every == 0:
            #     save_ckpt(itr)

            if itr % args.print_every==0 and (args.mog_comps is not None or args.nice):
                if (args.dataset in TOY_DSETS and itr % args.viz_every == 0) or args.dataset not in TOY_DSETS:
                    lps = []
                    for x_d, _ in test_loader:
                        if args.dataset in TOY_DSETS:
                            x_d = utils.toy_data.inf_train_gen(args.dataset, batch_size=args.batch_size)
                            x_d = torch.from_numpy(x_d).float().to(device)
                        else:
                            x_d = x_d.to(device)

                        lp = logp_net(x_d)
                        lps.append(lp)
                    lps = torch.cat(lps)
                    lp = lps.mean().item()
                    test_lps.append(lp)
                    test_lps_argmax = max(enumerate(test_lps), key=itemgetter(1))[0]
                    utils.print_log("Epoch {}, logp(x) {:.4f}, best logp(x) {:.4f}".
                                    format(epoch, test_lps[-1], test_lps[test_lps_argmax]), args)
                    plt.clf()
                    plt.plot(test_lps)
                    plt.savefig("{}/lp.png".format(args.save_dir))

            if args.clf and itr % args.eval_every == 0:
                eval_itrs.append(itr)

                # evaluate the accuracy of the model at the end of the epoch on the test set
                train_accs.append(train_acc)
                accs = []
                y_ds = []
                y_preds = []
                all_class_probs = []
                logp_net.eval()
                for x_d_, y_d_ in test_loader:
                    if args.dataset in TOY_DSETS:
                        x_d_ = utils.toy_data.inf_train_gen(args.dataset, batch_size=args.batch_size)
                        x_d_ = torch.from_numpy(x_d_).float().to(device)
                    else:
                        x_d_ = x_d_.to(device)
                    y_d_ = y_d_.to(device)

                    _, ld_logits = logp_net(x_d_, return_logits=True)

                    chosen = ld_logits.max(1).indices
                    acc = (chosen == y_d_).float()

                    y_preds.append(chosen)

                    class_probs = torch.nn.functional.softmax(ld_logits.detach(), dim=1)
                    y_ds.append(y_d_)
                    all_class_probs.append(class_probs)

                    accs.append(acc)

                y_ds = torch.cat(y_ds, dim=0)
                y_preds = torch.cat(y_preds, dim=0)

                all_class_probs = torch.cat(all_class_probs, dim=0)
                if args.num_classes == 2:
                    auc = roc_auc_score(y_true=y_ds.cpu(), y_score=all_class_probs[:, 1].cpu())
                aucs.append(auc)

                if args.num_classes == 2:
                    brier = brier_score_loss(y_true=y_ds.cpu(), y_prob=all_class_probs[:, 1].cpu())
                else:
                    targets = torch.zeros((y_ds.size(0), args.num_classes)).to(device)
                    targets.scatter_(1, y_ds[:, None], 1)
                    brier = brier_score_loss_multi(y_true=targets, y_prob=all_class_probs).cpu()
                briers.append(brier)

                test_accs.append(torch.cat(accs).mean().item())

                test_accs_argmax = max(enumerate(test_accs), key=itemgetter(1))[0]
                aucs_argmax = max(enumerate(aucs), key=itemgetter(1))[0]
                briers_argmin = min(enumerate(briers), key=itemgetter(1))[0]
                utils.print_log("eval itr {}, "
                                "acc {:.4f} auc {:.4f}, brier {:.4f}, "
                                "best acc {:.4f} (auc {:.4f}) (brier {:.4f}) (itr {}), "
                                "best auc {:.4f} (acc {:.4f}) (brier {:.4f}) (itr {}), "
                                "best brier {:.4f} (acc {:.4f}) (auc {:.4f}) (itr {}) ".
                                format(itr,
                                       test_accs[-1], aucs[-1], briers[-1],
                                       max(test_accs), aucs[test_accs_argmax], briers[test_accs_argmax], eval_itrs[test_accs_argmax],
                                       max(aucs), test_accs[aucs_argmax], briers[aucs_argmax], eval_itrs[aucs_argmax],
                                       min(briers), test_accs[briers_argmin], aucs[briers_argmin], eval_itrs[briers_argmin]), args)

                plt.clf()
                plt.plot(eval_itrs, train_accs, label="train")
                plt.plot(eval_itrs, test_accs, label="test")
                plt.savefig("{}/acc.png".format(args.save_dir))

                is_max = test_accs_argmax == len(test_accs) - 1

                if is_max:
                    # save model weights and plot calibration for best performing model

                    save_ckpt(itr, overwrite=True, prefix="best")

                    plt.clf()
                    if args.num_classes == 2:
                        fracpos, mean_pred = calibration_curve(y_true=y_ds.cpu(),
                                                               y_prob=all_class_probs[:, 1].cpu(), n_bins=10)
                    else:
                        fracpos, mean_pred = calibration_curve(y_true=(y_ds == y_preds).cpu(),
                                                               y_prob=all_class_probs.max(1)[0].cpu(), n_bins=10)
                    plt.plot(mean_pred, fracpos, "s-")
                    plt.xlabel("Mean predicted value")
                    plt.ylabel("Fraction of positives")
                    plt.ylim([-.05, 1.05])
                    plt.plot([0, 1], [0, 1], "k:")
                    plt.savefig("{}/cal.png".format(args.save_dir))

                logp_net.train()

def test_plot(args):
    test_dir = "{}/{}".format(args.save_dir, "test")
    utils.makedirs(test_dir)
    logp_net, g = get_models(args)
    pkl_path='/data/juicefs_sharing_data/11163512/code/VERA/results/toy/25gaussians/1697682650/save_model/_180000.pt'
    ckpt = torch.load(pkl_path)
    logp_net.load_state_dict(ckpt["model"]["logp_net"])
    g.load_state_dict(ckpt["model"]["g"])
    g.eval()
    g.to(device)
    logp_net.eval()
    logp_net.to(device)
    points = utils.toy_data.inf_train_gen(args.dataset, batch_size=1000)
    #points= torch.from_numpy(x_d).float().to(device)

    samples,_ = g.sample(1000)

    samples = samples.detach().cpu().numpy()
    #points = points.detach().cpu().numpy()
    plt.clf()
    # ax = plt.subplot(1, 1, 1, aspect="equal", title='gen')
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(samples[:, 0], samples[:, 1], s=1)
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.xticks([])
    plt.yticks([])
    fig.savefig("%s/gene_%s.png" % (test_dir, args.dataset), bbox_inches='tight', pad_inches=0.03)
    plt.close()

    # ax.imshow(vmax=1.5,vmin=-1.5)
    # ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.xticks([])
    plt.yticks([])
    fig.savefig("%s/real_%s.png" % (test_dir, args.dataset), bbox_inches='tight', pad_inches=0.03)
    plt.close()
    # #ax.imshow(vmax=1.5, vmin=-1.5)
    # self.disc.cpu()
    fig = plt.figure(figsize=(2, 2))
    npts = 100
    side = np.linspace(-3, 3, npts)
    side2 = np.linspace(-3, 3, npts)
    # ax = plt.subplot(1, 4, 3, aspect="equal")
    x = np.asarray(np.meshgrid(side, side2)).transpose(1, 2, 0).reshape((-1, 2))
    x = torch.from_numpy(x).type(torch.float32).to(device)
    logpx = logp_net(x).squeeze()

    logpx = logpx
    logpx = logpx - logpx.logsumexp(0)
    # logpx = logpx - logpx.mean(0)
    px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
    px = px / px.sum()
    plt.imshow(px, origin='lower', aspect="auto")
    plt.xticks([])
    plt.yticks([])
    fig.savefig("%s/pr_%s.png" % (test_dir, args.dataset), bbox_inches='tight', pad_inches=0.03)
    plt.close()

def test_generation(args,method='VERA'):
    test_dir = "{}/{}".format(args.save_dir, "test")
    utils.makedirs(test_dir)
    logp_net, g = get_models(args)
    pkl_path = '/data/juicefs_sharing_data/11163512/code/VERA/results/1696661701/save_model/_150000.pt'
    ckpt = torch.load(pkl_path)
    logp_net.load_state_dict(ckpt["model"]["logp_net"])
    g.load_state_dict(ckpt["model"]["g"])
    g.eval()
    g.to(device)
    logp_net.eval()
    logp_net.to(device)

    batch_size = 16
    #with torch.no_grad():

    exact_sample = logp_net.sample(batch_size)
    fake_sample, h_g = g.sample(batch_size, requires_grad=True)



    fake_sample=fake_sample.view(batch_size,*args.data_size)
    torchvision.utils.save_image(F.sigmoid(fake_sample).view(batch_size, 1, 28, 28),
            os.path.join(test_dir, '{}_generate.png'.format(method)),nrow=8,)
    #exact_sample = torch.cat([netD.sample(32, t0), netD.sample(32, t0 + 1)], 0)
    exact_sample = exact_sample.view(batch_size, *args.data_size)
    torchvision.utils.save_image(F.sigmoid(exact_sample),
                                 os.path.join(test_dir, '{}_exact.png'.format(method)),nrow=8,)


"""
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export PYTHONPATH=./
        python ./train.py 

    :return:
    """
if __name__ == "__main__":
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser("Energy Based Models")

    # logging
    parser.add_argument("--log_file", type=str, default="log.txt")

    # data
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=list(TOY_DSETS) + list(TAB_DSETS) +
                                ["mnist", "stackmnist", "cifar10", "svhn"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--unit_interval", action="store_true")
    parser.add_argument("--logit", action="store_true")
    parser.add_argument("--nice", action="store_true")
    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument('--img_size', type=int)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--glr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.)
    parser.add_argument("--beta2", type=float, default=.9)
    parser.add_argument("--labels_per_class", type=int, default=0,
                        help="number of labeled examples per class, if zero then use all labels (no SSL)")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--n_epochs", type=int, default=180000)
    parser.add_argument("--sgld_steps", type=int, default=100)
    parser.add_argument('--mog_comps', type=int, default=None, help="Mixture of gaussians.")
    parser.add_argument("--g_feats", type=int, default=128)
    parser.add_argument("--e_iters", type=int, default=1)
    parser.add_argument("--g_iters", type=int, default=1)
    parser.add_argument("--decay_epochs", nargs="+", type=float, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=0.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--warmup_iters", type=float, default=0,
                        help="number of iterations to warmup the LR")

    # model
    parser.add_argument("--seed", type=int, default=1232)
    parser.add_argument("--h_dim", type=int, default=300)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--norm", type=str, default='batch', choices=[None, "batch", "group", "instance", "layer"])
    parser.add_argument("--no_g_batch_norm", action="store_true")

    parser.add_argument("--resnet", action="store_true", help="Use resnet architecture.")
    parser.add_argument("--wide_resnet", action="store_true", help="Use wide_resnet architecture")
    parser.add_argument("--thicc_resnet", action="store_true", help="Use 28-10 architecture")
    parser.add_argument("--max_sigma", type=float, default=.3)
    parser.add_argument("--min_sigma", type=float, default=.01)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--generator_type", type=str, default="vera", choices=["verahmc", "vera"])
    parser.add_argument("--clf_only", action="store_true", help="Only do classification")
    parser.add_argument("--jem", action="store_true", default=False, help="Classification and JEM training")
    parser.add_argument("--maximum_likelihood", action="store_true", default=False, help="ML baseline")
    parser.add_argument("--ssm", action="store_true", default=False, help="Sliced Score Matching baseline")

    # VAT baseline
    parser.add_argument("--vat", action="store_true", default=False, help="Run VAT instead of JEM")
    parser.add_argument("--vat_weight", type=float, default=1.0)
    parser.add_argument("--vat_eps", type=float, default=3.0)

    # JEM baseline
    parser.add_argument("--jem_baseline", action="store_true", default=False, help="Run original JEM")
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--sgld_lr", type=float, default=None)
    parser.add_argument("--sgld_std", type=float, default=.01)
    parser.add_argument("--reinit_freq", type=float, default=.05)

    # loss weighting
    parser.add_argument("--ent_weight", type=float, default=0.1)
    parser.add_argument("--clf_weight", type=float, default=1.)
    parser.add_argument("--clf_ent_weight", type=float, default=0.)
    parser.add_argument("--mcmc_lr", type=float, default=.02)
    parser.add_argument("--post_lr", type=float, default=.02)

    # regularization
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--p_control", type=float, default=0.0)
    parser.add_argument("--n_control", type=float, default=0.0)
    parser.add_argument("--pg_control", type=float, default=0.1)

    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./results')
    parser.add_argument("--ckpt_path", type=str, default='')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--save_every", type=int, default=30000, help="Saving models for evaluation")
    parser.add_argument("--eval_every", type=int, default=200, help="Evaluating models on validation set")
    parser.add_argument("--print_every", type=int, default=10000, help="Iterations between print")
    parser.add_argument("--viz_every", type=int, default=10000, help="Iterations between visualization")
    parser.add_argument("--load_path", type=str, default=None)

    args = parser.parse_args()

    if args.img_size is not None and args.img_size not in (32, 64):
        raise ValueError

    if args.sgld_lr is None:
        args.sgld_lr = args.sgld_std ** 2. / 2.

    if args.dataset in TOY_DSETS:
        args.data_dim = 2
        args.data_size = (2, )
    elif args.dataset == "HEPMASS":
        args.data_dim = 15
        args.num_classes = 2
    elif args.dataset == "HUMAN":
        args.data_dim = 523
        args.num_classes = 6
    elif args.dataset == "CROP":
        args.data_dim = 174
        args.num_classes = 7
    elif args.dataset == "mnist":
        args.num_classes = 10
        if args.img_size:
            args.data_dim = args.img_size ** 2
            args.data_size = (1, args.img_size, args.img_size)
        else:
            args.data_dim = 784
            args.data_size = (1, 28, 28)
    elif args.dataset == "stackmnist":
        args.num_classes = 1000
        if args.img_size:
            args.data_dim = 3 * args.img_size ** 2
            args.data_size = (3, args.img_size, args.img_size)
        else:
            args.data_dim = 784 * 3
            args.data_size = (3, 28, 28)
    elif args.dataset == "svhn" or args.dataset == "cifar10":
        args.num_classes = 10
        args.data_dim = 32 * 32 * 3
        args.data_size = (3, 32, 32)
    elif args.dataset == "cifar100":
        args.num_classes = 100
        args.data_dim = 32 * 32 * 3
        args.data_size = (3, 32, 32)
    else:
        raise ValueError

    if args.dataset in TAB_DSETS:
        args.data_size = (args.data_dim, )

    args.ssl = args.labels_per_class > 0
    assert not args.ssl or (args.jem or args.vat), "SSL implies JEM or VAT"

    assert not args.vat or args.jem, "VAT implies JEM"

    args.clf = args.ssl or args.jem or args.clf_only or args.vat
    assert not args.clf or (args.jem or args.vat), "Classification implies JEM or VAT"

    def strictly_increasing(lst):
        """
        Check if lst is strictly increasing.
        """
        return all(x < y for x, y in zip(lst[:-1], lst[1:]))

    assert strictly_increasing(args.decay_epochs), "Decay epochs should be strictly increasing"
    args.save_dir = os.path.join(args.save_dir + '/{}'.format(args.dataset)+'/%03d' % int(time.time()))
    utils.makedirs(args.save_dir)

    torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    utils.print_log('Using {} GPU(s).'.format(torch.cuda.device_count()), args)

    with open("{}/args.txt".format(args.save_dir), 'w') as f:
        json.dump(args.__dict__, f, indent=4, sort_keys=True)


    #test_plot(args)
    test_generation(args)
