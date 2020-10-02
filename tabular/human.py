
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

        # import matplotlib.pyplot as plt
        # print(labels_train)
        # plt.hist(labels_train)
        # plt.show()
        #
        # plt.clf()
        # #print(labels_test)
        # plt.hist(labels_test)
        # plt.show()
        # 1/0

        #labels_train = data_train[data_train.columns[0]].values
        #labels_test = data_test[data_test.columns[0]].values

        self.label_names = {
            0: "WALKING",
            1: "WALKING_UPSTAIRS",
            2: "WALKING_DOWNSTAIRS",
            3: "SITTING",
            4: "STANDING",
            5: "LAYING",
        }

        #data_train = data_train.drop(data_train.columns[0], axis=1)
        #data_test = data_test.drop(data_test.columns[0], axis=1)

        # Because the data set is messed up!
        #data_test = data_test.drop(data_test.columns[-1], axis=1)

        data_train = data_train.to_numpy()
        print(data_train.shape)
        data_test = data_test.to_numpy()

        #print(data_train.mean(0), data_train.std(0))


        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            #print(max_count)
            if max_count > 5:
                features_to_remove.append(i)
            i += 1

        print(data_train.shape)
        data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        print(data_train.shape)
        data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]
        print(data_train.shape, labels_train.shape)
        print(data_train[0])
        return {"trn": (data_train, labels_train),
                "tst": (data_test, labels_test),
                "shuffle": True}


if __name__ == "__main__":
    x = HUMAN("../data/HUMAN")
    counts = np.bincount(x.trn.y)
    print(counts.max() / counts.sum())
