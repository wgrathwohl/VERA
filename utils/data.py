"""
Utilities for data.
"""

import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets

from utils import toy_data
from .toy_data import TOY_DSETS
from tabular import TAB_DSETS


def logit(x, alpha=0.):
    x = x * (1. - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1. - x)


def get_data(args):
    """
    Get data.
    """
    if args.unit_interval:
        post_trans = [lambda x: x]
        post_trans_inv = lambda x: x
    elif args.logit:
        post_trans = [lambda x: (x + (torch.rand_like(x) - 0.5) / 256).clamp(1e-3, 1-1e-3), logit]
        post_trans_inv = lambda x: x.sigmoid()
    else:
        post_trans = [lambda x: 2 * x - 1]
        post_trans_inv = lambda x: (x + 1) / 2
    if args.dataset in TOY_DSETS:
        data = torch.from_numpy(toy_data.inf_train_gen(args.dataset, batch_size=args.batch_size)).float()
        dset = TensorDataset(data, data)  # add data as "labels" to match api for image dsets
        dload = DataLoader(dset, args.batch_size, True, drop_last=True)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(x, p, normalize=False, nrow=sqrt(x.size(0)))
        return dload, dload, plot
    elif args.dataset in TAB_DSETS:
        dset = TAB_DSETS[args.dataset](seed=args.seed)

        tr_dataset = TensorDataset(torch.tensor(dset.trn.x), torch.tensor(dset.trn.y))
        te_dataset = TensorDataset(torch.tensor(dset.val.x), torch.tensor(dset.val.y))

        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)

        return tr_dload, te_dload, None
    elif args.dataset == "mnist":
        if args.img_size is not None:
            pre_trans = [transforms.Resize(args.img_size),
                         transforms.ToTensor()]
        else:
            pre_trans = [transforms.ToTensor()]
            post_trans += [lambda x: x.view(-1)]

        tr_dataset = datasets.MNIST("./data",
                                    transform=transforms.Compose(pre_trans +
                                                                 post_trans),
                                    download=True)
        te_dataset = datasets.MNIST("./data", train=False,
                                    transform=transforms.Compose(pre_trans + post_trans),
                                    download=True)


        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: \
            torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))

        return tr_dload, te_dload, plot
    elif args.dataset == "stackmnist":
        if args.img_size is not None:
            pre_trans = [transforms.Resize(args.img_size), transforms.ToTensor()]
        else:
            pre_trans = [transforms.ToTensor()]
            post_trans += [lambda x: x.view(-1)]
        tr_dataset = datasets.MNIST("./data",
                                    transform=transforms.Compose(pre_trans +
                                                                 post_trans),
                                    download=True)

        def dataset_to_tensor(dataset):
            """
            Convert dataset to tensor (in particular apply resizing transformations).
            """
            loader = DataLoader(dataset, batch_size=len(dataset))
            return next(iter(loader))

        def stack_mnist(dataset):
            """
            Stack 3 MNIST images along 3 channels.
            """
            x, y = dataset_to_tensor(dataset)
            np.random.seed(args.seed)  # seed so we always train on the same stackmnist
            ids = np.random.randint(0, x.shape[0], size=(x.shape[0], 3))
            X_training = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
            Y_training = torch.zeros(x.shape[0])
            for i in range(ids.shape[0]):
                cnt = 0
                for j in range(ids.shape[1]):
                    xij = x[ids[i, j]]
                    X_training[i, j] = xij
                    cnt += y[ids[i, j]] * (10**j)
                Y_training[i] = cnt
                if i % 10000 == 0:
                    print('i: {}/{}'.format(i, ids.shape[0]))

            return TensorDataset(X_training, Y_training)

        tr_dataset = stack_mnist(tr_dataset)

        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)

        def plot(p, x):
            """
            Unstack images for plotting.
            """
            x = torch.cat((x[:, 0], x[:, 1], x[:, 2]), dim=0)[:, None]
            sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
            return torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))

        return tr_dload, None, plot

    elif args.dataset == "stack4mnist":
        if args.img_size is not None:
            pre_trans = [transforms.Resize(args.img_size), transforms.ToTensor()]
        else:
            pre_trans = [transforms.ToTensor()]
            post_trans += [lambda x: x.view(-1)]
        tr_dataset = datasets.MNIST("./data",
                                    transform=transforms.Compose(pre_trans +
                                                                 post_trans),
                                    download=True)

        def dataset_to_tensor(dataset):
            """
            Convert dataset to tensor (in particular apply resizing transformations).
            """
            loader = DataLoader(dataset, batch_size=len(dataset))
            return next(iter(loader))

        def stack_mnist(dataset):
            """
            Stack 4 MNIST images along 4 channels.
            """
            x, y = dataset_to_tensor(dataset)
            n_data = 100 * 10 ** 3
            np.random.seed(args.seed)  # seed so we always train on the same stackmnist
            ids = np.random.randint(0, x.shape[0], size=(n_data, 4))
            X_training = torch.zeros(n_data, 4, x.shape[2], x.shape[3])
            Y_training = torch.zeros(n_data)
            for i in range(ids.shape[0]):
                cnt = 0
                for j in range(ids.shape[1]):
                    xij = x[ids[i, j]]
                    X_training[i, j] = xij
                    cnt += y[ids[i, j]] * (10**j)
                Y_training[i] = cnt
                if i % 10000 == 0:
                    print('i: {}/{}'.format(i, ids.shape[0]))

            return TensorDataset(X_training, Y_training)

        tr_dataset = stack_mnist(tr_dataset)

        _, tr_labels = tr_dataset.tensors
        tr_label_counts = torch.Tensor(np.bincount(tr_labels, minlength=10000))

        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)

        def plot(p, x):
            """
            Unstack images for plotting.
            """
            x = torch.cat((x[:, 0], x[:, 1], x[:, 2], x[:, 3]), dim=0)[:, None]
            sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
            return torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))

        return tr_dload, None, plot, tr_label_counts

    elif args.dataset == "svhn":
        if args.data_aug:
            augs = [transforms.Pad(4, padding_mode="reflect"), transforms.RandomCrop(32)]
            print("using data augmentation")
        else:
            augs = []
        tr_dataset = datasets.SVHN("./data",
                                    transform=transforms.Compose(augs +
                                                                 [transforms.ToTensor()] +
                                                                 post_trans),
                                    download=True)
        te_dataset = datasets.SVHN("./data", split='test',
                                    transform=transforms.Compose([transforms.ToTensor()] +
                                                                 post_trans),
                                    download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))
        return tr_dload, te_dload, plot

    elif args.dataset == "cifar10":
        if args.data_aug:
            augs = [transforms.Pad(4, padding_mode="reflect"),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip()]
            print("using data augmentation")
        else:
            augs = []
        tr_dataset = datasets.CIFAR10("./data",
                                      transform=transforms.Compose(augs +
                                                                   [transforms.ToTensor()] +
                                                                   post_trans),
                                      download=True)
        te_dataset = datasets.CIFAR10("./data", train=False,
                                      transform=transforms.Compose([transforms.ToTensor()] +
                                                                   post_trans),
                                      download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))
        return tr_dload, te_dload, plot

    elif args.dataset == "cifar100":
        if args.data_aug:
            augs = [transforms.Pad(4, padding_mode="reflect"),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip()]
            print("using data augmentation")
        else:
            augs = []
        tr_dataset = datasets.CIFAR100("./data",
                                       transform=transforms.Compose(augs +
                                                                   [transforms.ToTensor()] +
                                                                   post_trans),
                                       download=True)
        te_dataset = datasets.CIFAR100("./data", train=False,
                                       transform=transforms.Compose([transforms.ToTensor()] +
                                                                   post_trans),
                                       download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))
        return tr_dload, te_dload, plot

    else:
        raise NotImplementedError
