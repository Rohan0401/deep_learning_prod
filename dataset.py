import numpy as np

class Dataset:
    def __init__(self, train, test, val_frac=0.5, shuffle=True, scale_func=None):
        split_idx = int (len (test['y']) * (1 - val_frac))
        self.test_x, self.valid_x = test['X'][:, :, :, :split_idx], test['X'][:, :, :, split_idx:]
        self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
        self.train_x, self.train_y = train['X'], train['y']
        # The SVHN dataset comes with lots of labels, but for the purpose of this exercise,
        # we will pretend that there are only 1000.
        # We use this mask to say which labels we will allow ourselves to use.
        self.label_mask = np.zeros_like (self.train_y)
        self.label_mask[0:1000] = 1

        self.train_x = np.rollaxis (self.train_x, 3)
        self.valid_x = np.rollaxis (self.valid_x, 3)
        self.test_x = np.rollaxis (self.test_x, 3)

        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        self.train_x = self.scaler (self.train_x)
        self.valid_x = self.scaler (self.valid_x)
        self.test_x = self.scaler (self.test_x)
        self.shuffle = shuffle

    def batches(self, batch_size, which_set="train"):
        x_name = which_set + "_x"
        y_name = which_set + "_y"

        num_examples = len (getattr (dataset, y_name))
        if self.shuffle:
            idx = np.arange (num_examples)
            np.random.shuffle (idx)
            setattr (dataset, x_name, getattr (dataset, x_name)[idx])
            setattr (dataset, y_name, getattr (dataset, y_name)[idx])
            if which_set == "train":
                dataset.label_mask = dataset.label_mask[idx]

        dataset_x = getattr (dataset, x_name)
        dataset_y = getattr (dataset, y_name)
        for ii in range (0, num_examples, batch_size):
            x = dataset_x[ii:ii + batch_size]
            y = dataset_y[ii:ii + batch_size]

            if which_set == "train":
                # When we use the data for training, we need to include
                # the label mask, so we can pretend we don't have access
                # to some of the labels, as an exercise of our semi-supervised
                # learning ability
                yield x, y, self.label_mask[ii:ii + batch_size]
            else:
                yield x, y