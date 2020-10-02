"""
Count the number of modes online captured from stackmnist using a MNIST classifier.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from mode_counting.mnist_classify import Net


class TransformTensorDataset(Dataset):
    """
    TensorDataset supporting Transforms.
    """
    def __init__(self, tensors, transform=None):
        super().__init__()
        self.tensors = tensors
        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            for transform in self.transform:
                x = transform(x)

        return x


def evaluate_model(model,
                   label_counts,
                   args,
                   device,
                   num_samples=100000,
                   batch_size=10000,
                   classifier_dir="mode_counting"):
    """
    Evaluate the number of modes from a model.
    """
    assert num_samples % batch_size == 0
    num_batches = num_samples // batch_size

    classifier = Net(args.img_size)
    classifier.to(device)
    classifier.load_state_dict(torch.load(os.path.join(classifier_dir, "mnist_cnn_%d.pt" % args.img_size),
                                          map_location=device))
    classifier.eval()  # evaluation mode, turn off dropout

    # classify
    def _classify_batch(x, ind):
        """
        Classify one layer of stackmnist.
        """
        return classifier(x[:, ind][:, None]).argmax(dim=1)

    def _kl(p, q):
        """
        Computes reverse KL between vectors p and q.
        """
        ent = p.log() - q.log()

        ent[~torch.isfinite(ent)] = 0
        return torch.dot(p, ent)

    def summarize_samples():
        """
        Count number of modes captured in a batch of samples.
        """
        counts = torch.zeros(10000)
        for _ in range(num_batches):
            with torch.no_grad():
                x_g, _ = model.sample(batch_size)
                x_g = x_g.view(x_g.size(0), *args.data_size)
                preds = sum(_classify_batch(x_g, i) * (10 ** i) for i in range(4))
            counts += np.bincount(preds.cpu().numpy(), minlength=10000)
        return (counts != 0).sum(), torch.Tensor((counts / counts.sum()).float())

    model_num_modes, model_label_dist = summarize_samples()

    data_num_modes = (label_counts != 0).sum()

    data_label_dist = label_counts / label_counts.sum()

    klqp = _kl(model_label_dist, data_label_dist)

    return data_num_modes, model_num_modes, klqp


# duplicated code is very bad umu
def evaluate_original_jem_model(sampler,
                   label_counts,
                   args,
                   device,
                   num_samples=100000,
                   batch_size=10000,
                   classifier_dir="mode_counting"):
    """
    Evaluate the number of modes from a model.
    """
    assert num_samples % batch_size == 0
    num_batches = num_samples // batch_size

    classifier = Net(args.img_size)
    classifier.to(device)
    classifier.load_state_dict(torch.load(os.path.join(classifier_dir, "mnist_cnn_%d.pt" % args.img_size),
                                          map_location=device))
    classifier.eval()  # evaluation mode, turn off dropout

    # classify
    def _classify_batch(x, ind):
        """
        Classify one layer of stackmnist.
        """
        return classifier(x[:, ind][:, None]).argmax(dim=1)

    def _kl(p, q):
        """
        Computes reverse KL between vectors p and q.
        """
        ent = p.log() - q.log()

        ent[~torch.isfinite(ent)] = 0
        return torch.dot(p, ent)

    def summarize_samples():
        """
        Count number of modes captured in a batch of samples.
        """
        counts = torch.zeros(10000)
        for _ in range(num_batches):
            x_g = sampler(batch_size).detach()
            with torch.no_grad():
                x_g = x_g.view(x_g.size(0), *args.data_size)
                preds = sum(_classify_batch(x_g, i) * (10 ** i) for i in range(4))
            counts += np.bincount(preds.cpu().numpy(), minlength=10000)
        return (counts != 0).sum(), torch.Tensor((counts / counts.sum()).float())

    model_num_modes, model_label_dist = summarize_samples()

    data_num_modes = (label_counts != 0).sum()

    data_label_dist = label_counts / label_counts.sum()

    klqp = _kl(model_label_dist, data_label_dist)

    return data_num_modes, model_num_modes, klqp
