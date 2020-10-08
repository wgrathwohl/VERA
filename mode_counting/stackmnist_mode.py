"""
Count the number of modes captured from stackmnist using a MNIST classifier.
"""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import utils.data
from models.get_models import get_models
from mode_counting.mnist_classify import Net
import utils


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


def main(parse_args):
    """
    Main function.
    """

    # load the trained MNIST classifier
    classifier = Net(parse_args.img_size)
    classifier.load_state_dict(torch.load("experiments/mnist_cnn_%d.pt" % parse_args.img_size,
                                          map_location=torch.device('cpu')))
    classifier.eval()  # evaluation mode, turn off dropout

    # paths to the models we'd like to evaluate
    eval_dirs = []

    eval_itrs = []

    def load_model(directory, itr,  return_p=False):
        """
        Load models given experiment directory and itr.
        """
        path = os.path.join("experiments", directory, "save_model", "{:06d}.pt".format(itr))

        # load arguments
        with open(os.path.join("experiments", directory, "args.txt"), 'r') as f:
            args = argparse.Namespace(**json.load(f))

        logp_net, g = get_models(args, log=False)

        ckpt = torch.load(path, map_location=torch.device('cpu'))

        logp_net.load_state_dict(ckpt["model"]["logp_net"])
        g.load_state_dict(ckpt["model"]["g"])

        # get true labels
        train_loader, _, _ = utils.data.get_data(args)
        label_counts = torch.zeros(1000)
        for _, label in train_loader:
            label_counts[label.long()] += 1

        label_counts /= label_counts.sum()

        if return_p:
            return logp_net, g, args, label_counts
        else:
            return logp_net, g, args

    for eval_dir, eval_itr in zip(eval_dirs, eval_itrs):
        # load model and config
        _, g, args, p = load_model(eval_dir, eval_itr, return_p=True)

        # get samples and resize them
        x_g, _ = g.sample(parse_args.num_samples, requires_grad=False)
        x_g = x_g.view(x_g.size(0), *args.data_size)
        x_g = TensorDataset(x_g)
        sample_loader = DataLoader(x_g, batch_size=parse_args.batch_size)

        # classify
        def classify_batch(x, ind):
            """
            Classify one layer of stackmnist.
            """
            return classifier(x[:, ind][:, None]).argmax(dim=1)

        def classify(loader):
            """
            Classify all examples in dataloader.
            """
            assert parse_args.num_samples % parse_args.batch_size == 0
            labels = []
            for i, (sample, ) in enumerate(loader):
                print(i, len(loader))
                label0 = classify_batch(sample, 0)
                label1 = classify_batch(sample, 1)
                label2 = classify_batch(sample, 2)
                label = label0 * 100 + label1 * 10 + label2
                labels.append(list(label))
            return torch.tensor(labels)

        def num_modes(loader, return_labels=False):
            """
            Count number of modes captured in a batch of samples.
            """
            label = classify(loader)
            if return_labels:
                return len(label.unique()), label
            else:
                return len(label.unique())

        def _kl(p, q):
            """
            Computes KL between vectors p and q.
            """
            ent1 = p.log() - q.log()
            ent2 = q.log() - p.log()

            ent1[~torch.isfinite(ent1)] = 0
            ent2[~torch.isfinite(ent2)] = 0
            return torch.dot(p, ent1), torch.dot(q, ent2)

        def kl_modes(loader):
            """
            Compute the KL between the true labels and the labels of samples.
            """
            labels = classify(loader)

            # compute the densities assigned to each label
            sample_counts = torch.zeros(1000)
            for label in labels:
                sample_counts[label] += 1
            sample_counts /= sample_counts.sum()

            klpq, klqp = _kl(p, sample_counts)

            return klpq, klqp

        sample_modes = num_modes(sample_loader)
        print(eval_dir, eval_itr, sample_modes)
        klpq, klqp = kl_modes(sample_loader)  # klpq is wrong, undefined!
        print(eval_dir, eval_itr, klqp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Counting Modes")

    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples from model")
    parser.add_argument("--batch_size", type=int, default=100, help="Batches for evaluating model")
    parser.add_argument("--img_size", type=int, default=64, help="Size of MNIST image")

    args = parser.parse_args()

    main(args)
