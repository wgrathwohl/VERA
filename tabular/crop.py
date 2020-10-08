"""
Data processing for CROP.
"""

import os

import numpy as np
import pandas as pd

from tabular import utils


class CROP(utils.Tabular):
    """
    The CROP data set.
    https://archive.ics.uci.edu/ml/datasets/Crop+mapping+using+fused+optical-radar+data+set
    """

    def __init__(self, data_root="./data/CROP", **kwargs):
        super().__init__(data_root, **kwargs)

    def load(self, path, drop_thresh=1.01):
        """
        Load the train and test sets, and split a validation set from train.
        """
        data = pd.read_csv(filepath_or_buffer=os.path.join(path, "WinnipegDataset.txt"))
        labels = data['label']
        labels = [l - 1 for l in labels]

        for i in range(7):
            n = [l for l in labels if l == i]
            print(i, len(n))
        X = data.to_numpy()[:, 1:]

        Xm, Xstd = X.mean(0), X.std(0)

        Xs = (X - Xm[None, :]) / Xstd[None, :]

        cov = np.cov(Xs.T)

        drop_inds = []
        for i in range(cov.shape[0]):
            for j in range(i + 1, cov.shape[0]):
                assert i != j
                if abs(cov[i, j]) > drop_thresh:
                    drop_inds.append(j)

        self.label_names = {i: "idk-{}".format(i) for i in range(7)}

        data = data.drop(data.columns[0], axis=1)
        all_keys = data.keys()[1:]
        keys_to_drop = [all_keys[ind] for ind in drop_inds]
        data = data.drop(columns=keys_to_drop)

        return {"trn": (data.to_numpy(), np.array(labels)), "shuffle": True}


if __name__ == "__main__":
    CROP("../data/CROP")
