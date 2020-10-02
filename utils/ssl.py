"""
Utilities for semi-supervised learning.
"""

import numpy as np
from torch.utils.data import Dataset


def cycle(loader):
    """
    Create infinite cycle through a generator.
    """
    while True:
        for data in loader:
            yield data


class DataSubset(Dataset):
    """
    Torch interface for labelled subset for semi-supervised learning.
    """
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


def labeled_subset(dataset, n_labels_per_class, seed=1234, n_class=10):
    """
    Labelled subset for semi-supervised learning.
    """
    np.random.seed(seed)
    inds = np.array(list(range(len(dataset))))
    np.random.shuffle(inds)
    labels = np.array([dataset[i][1] for i in inds])
    # unique, counts = np.unique(labels, return_counts=True)
    # counts = 100 * counts / counts.sum()
    # counts.sort()
    # print(counts)
    # print(counts.sum())
    chosen_inds = []
    class_counts = []
    for c in range(n_class):
        class_inds = inds[labels == c]
        # print(class_inds)
        # print("found {} examples with class {}".format(len(class_inds), c))
        chosen = class_inds[:n_labels_per_class]
        class_counts.append(len(chosen))
        chosen_inds.extend(chosen)
    # print("chose {} inds".format(len(chosen_inds)))
    class_counts = np.array(class_counts)
    class_counts = 100 * class_counts / class_counts.sum()
    # print(class_counts)
    # print(class_counts.max())
    # print(class_counts.min())
    # print(class_counts.sum())
    ds = DataSubset(dataset, chosen_inds)
    return ds
