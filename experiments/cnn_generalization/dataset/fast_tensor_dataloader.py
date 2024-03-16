import torch


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(self, dataset, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        self.dataset = dataset
        assert all(
            t.shape[0] == self.dataset.tensors[0].shape[0] for t in self.dataset.tensors
        )
        self.tensors = self.dataset.tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.device = self.tensors[0].device
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.batch_size == self.dataset_len:
            # check if this is the first full batch
            if self.i == 0:
                # raise counter
                self.i = 1
                return self.tensors
            else:
                raise StopIteration
        else:
            if self.i >= self.dataset_len:
                raise StopIteration
            if self.indices is not None:
                indices = self.indices[self.i : self.i + self.batch_size]
                batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
            else:
                batch = tuple(
                    t[self.i : self.i + self.batch_size] for t in self.tensors
                )
            self.i += self.batch_size
            return batch

    def __len__(self):
        return self.n_batches
