"""
Data processing for HEPMASS.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

from tabular import utils


class HEPMASS(utils.Tabular):
    """
    The HEPMASS data set.
    http://archive.ics.uci.edu/ml/datasets/HEPMASS
    """

    def __init__(self, data_root="./data/HEPMASS", **kwargs):
        super().__init__(data_root, **kwargs)

    def load(self, path):
        """
        Load the train and test sets, and split a validation set from train.
        """
        data_train = pd.read_csv(filepath_or_buffer=os.path.join(path, "1000_train.csv"), index_col=False)
        data_test = pd.read_csv(filepath_or_buffer=os.path.join(path, "1000_test.csv"), index_col=False)

        labels_train = data_train[data_train.columns[0]].values
        labels_test = data_test[data_test.columns[0]].values

        self.label_names = {0: "background", 1: "signal"}

        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test.drop(data_test.columns[0], axis=1)

        # Because the data set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)

        data_train = data_train.to_numpy()
        data_test = data_test.to_numpy()

        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

        return {"trn": (data_train, labels_train),
                "tst": (data_test, labels_test)}


if __name__ == "__main__":
    HEPMASS("../data/HEPMASS")
