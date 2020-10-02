"""
Utilities for tabular data.
"""

import numpy as np


class Data:
    """
    Data class interface
    """

    def __init__(self, data):

        self.x, self.y = data
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.int64)
        self.n = self.x.shape[0]


class Tabular:
    """
    Abstract class for tabular data.
    """
    def __init__(self, path, seed=1234):
        self.seed = seed

        trn, val, tst = self.load_and_split(path)

        self.trn = Data(trn)
        self.val = Data(val)
        self.tst = Data(tst)

        self.n_dims = self.trn.x.shape[1]

    @property
    def label_names(self):
        """
        Get the names for the labels.
        """
        return self._label_names

    @label_names.setter
    def label_names(self, value):
        self._label_names = value

    @property
    def num_classes(self):
        """
        Get the number of labels.
        """
        return len(self._label_names)

    def load(self, path):
        """
        Load the data from a file into a dictionary of matrices, one for trn, val, test.
        Each split returned should be (x, y), both numpy arrays.
        """
        raise NotImplementedError

    def load_and_split(self, path):
        """
        Load and split the data.
        """
        return Tabular._standardize_data(*self._split_data(**self.load(path)))

    def _split_data(self, trn, val=None, tst=None, val_ratio=.1, tst_ratio=.1, shuffle=False):
        """
        Split the data into train, val, test.
        """
        assert tst is not None or val is None, "tst is None implies val is None"

        def _check_lengths(dat):
            if dat is not None:
                x, y = dat
                n, m = x.shape
                print(y.shape, x.shape)
                assert y.shape == (n, )

        _check_lengths(trn)
        _check_lengths(val)
        _check_lengths(tst)

        def _check_shapes():
            n, m = trn[0].shape

            def _check(dat):
                if dat is not None:
                    assert len(dat[0].shape) == 2
                    assert dat[0].shape[1] == m

            _check(val)
            _check(tst)

        _check_shapes()

        if shuffle:
            np.random.seed(self.seed)
            inds = np.arange(trn[1].shape[0])
            np.random.shuffle(inds)
            trn = trn[0][inds], trn[1][inds]

        def _split(dat, ratio):

            N = len(dat[0])
            split_ind = int(N * ratio)

            def _index(slice_):
                return dat[0][slice_], dat[1][slice_]

            return _index(slice(-split_ind)), _index(slice(-split_ind, N))

        if tst is None:
            # split into train and test
            trn, tst = _split(trn, tst_ratio)

        if val is None:
            # split it off from the train set.
            trn, val = _split(trn, val_ratio)

        return trn, val, tst

    @staticmethod
    def _standardize_data(trn, val, tst):
        """
        All data should be standardized according to values computed on the training set.
        """
        mean, std = trn[0].mean(0), trn[0].std(0)

        def _standardize(data):
            x, y = data
            return ((x - mean[None]) / std[None], y)
        return _standardize(trn), _standardize(val), _standardize(tst)
