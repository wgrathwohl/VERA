"""
Data processing for HUMAN.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

from tabular import utils


class HUMAN(utils.Tabular):
    """
    The Human activity recognition data set.
    https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
    """

    def __init__(self, data_root="./data/HUMAN", **kwargs):
        super().__init__(data_root, **kwargs)

    def load(self, path):
        """
        Load the train and test sets, and split a validation set from train.
        """
        data_train = pd.read_csv(filepath_or_buffer=os.path.join(path, "train", "X_train.txt"),
                                 delim_whitespace=True, names=[str(x) for x in range(561)])
        data_test = pd.read_csv(filepath_or_buffer=os.path.join(path, "test", "X_test.txt"),
                                delim_whitespace=True, names=[str(x) for x in range(561)])

        with open(os.path.join(path, "train", "y_train.txt"), 'r') as f:
            labels_train = np.array([int(l.strip()) - 1 for l in f])

        with open(os.path.join(path, "test", "y_test.txt"), 'r') as f:
            labels_test = np.array([int(l.strip()) - 1 for l in f])

        self.label_names = {
            0: "WALKING",
            1: "WALKING_UPSTAIRS",
            2: "WALKING_DOWNSTAIRS",
            3: "SITTING",
            4: "STANDING",
            5: "LAYING",
        }

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
                "tst": (data_test, labels_test),
                "shuffle": True}


if __name__ == "__main__":
    HUMAN("../data/HUMAN")
