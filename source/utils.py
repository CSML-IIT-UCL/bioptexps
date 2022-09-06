import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np


class Logger:
    def __init__(self, writer=None, console=True):
        self.console = console
        if writer is not None:
            self.writer = writer

    def add_scalar(self, tag, val, step=None, console=False):
        if self.writer:
            self.writer.add_scalar(tag, val, step)
        console = True if console else self.console
        if console:
            print(f"{tag}= {val}")


class CustomTensorIterator:
    def __init__(self, tensor_list, batch_size, **loader_kwargs):
        exclude_args = ["num_workers", "pin_memory"]
        loader_kwargs = {
            k: v for k, v in loader_kwargs.items() if k not in exclude_args
        }

        self.loader = FastTensorDataLoader(
            *tensor_list, batch_size=batch_size, **loader_kwargs
        )
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            idx = next(self.iterator)
        return idx


class CustomTensorIteratorV2:
    """
    hack to have fast minibatches with sparse tensor (loads some of them in memory and cycles through)
    """

    def __init__(self, tensor_list, batch_size, **loader_kwargs):
        kwargs = {"num_workers": 2, "pin_memory": True}
        loader_kwargs = loader_kwargs if loader_kwargs is not None else kwargs
        loader_kwargs["shuffle"] = True  # doesnt work with this set to true,
        self.loader = DataLoader(
            TensorDataset(*tensor_list), batch_size=batch_size, **loader_kwargs
        )
        self.iterator = iter(self.loader)

        self.minibatches = []
        self.n_batches = 100
        self.idx = 0
        n_stop = 0
        while n_stop < 100:
            try:
                idx = next(self.iterator)
                self.minibatches.append(idx)
            except StopIteration:
                n_stop += 1

        self.rand_indices = torch.randperm(len(self.minibatches))

    def __next__(self, *args):
        if self.idx >= len(self.minibatches):
            self.idx = 0
            self.rand_indices = torch.randperm(len(self.minibatches))

        idx = self.minibatches[self.idx]
        self.idx += 1
        return idx


class CustomTensorIteratorV3:
    def __init__(self, tensor_list, batch_size, **loader_kwargs):
        kwargs = {"num_workers": 2, "pin_memory": True}
        loader_kwargs = loader_kwargs if loader_kwargs is not None else kwargs
        self.loader = DataLoader(
            TensorDataset(*tensor_list), batch_size=batch_size, **loader_kwargs
        )
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            idx = next(self.iterator)
        return idx


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = torch.tensor(list(range(self.dataset_len)))
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration

        indices = self.indices[self.i : self.i + self.batch_size]

        if self.tensors[0].is_sparse:
            # still slow
            batch = tuple(
                torch.cat([t[i].unsqueeze(0) for i in indices]) for t in self.tensors
            )
            # batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)

        else:
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)

        self.i += self.batch_size
        return batch

    # def __iter__(self):
    #     if self.shuffle:
    #         r = torch.randperm(self.dataset_len)
    #         self.tensors = [t[r] for t in self.tensors]
    #     self.i = 0
    #     return self
    #
    # def __next__(self):
    #     if self.i >= self.dataset_len:
    #         raise StopIteration
    #     batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
    #     self.i += self.batch_size
    #     return batch
    #
    # def __len__(self):
    #     return self.n_batches


def clean_dataset(X, y, n_samples, n_features, seed=0):
    """Reduce the number of features and / or samples.
    And remove lines or columns with only 0.
    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Design matrix.
    y: ndarray, shape (n_samples,)
        Observation vector.
    n_samples: int
        Number of samples to keep
    n_features: int
        Number of features to keep
    seed: int
        Seed for the random selection of the samples or features
    """
    np.random.seed(seed)
    idx = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
    feats = np.random.choice(X.shape[1], min(n_features, X.shape[1]), replace=False)
    X = X[idx, :]
    X = X[:, feats]
    y = y[idx]

    bool_to_keep = scipy.sparse.linalg.norm(X, axis=0) != 0
    X = X[:, bool_to_keep]
    bool_to_keep = scipy.sparse.linalg.norm(X, axis=1) != 0
    X = X[bool_to_keep, :]
    y = y[bool_to_keep]

    ypd = pd.DataFrame(y)
    bool_to_keep = ypd.groupby(0)[0].transform(len) > 2
    ypd = ypd[bool_to_keep]
    X = X[bool_to_keep.to_numpy(), :]
    y = y[bool_to_keep.to_numpy()]

    bool_to_keep = scipy.sparse.linalg.norm(X, axis=0) != 0
    X = X[:, bool_to_keep]
    bool_to_keep = scipy.sparse.linalg.norm(X, axis=1) != 0
    X = X[bool_to_keep, :]
    y = y[bool_to_keep]

    return X, y


def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.

    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU

    eps: float
      radius of l-1 ball to project onto

    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original

    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.

    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


def set_seed(torch, np, seed: int, is_deterministic: bool = True):
    # set the seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return 0 if len(memory_available) == 0 else np.argmax(memory_available)


def set_seed(torch, np, seed: int, is_deterministic: bool = True):
    # set the seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def bettercycle(iterable):
    while True:
        for i in iterable:
            yield i
