from libsvmdata.datasets import fetch_libsvm
from sklearn.datasets import fetch_20newsgroups_vectorized
from torchvision import datasets
from sklearn.model_selection import train_test_split


def load_mnist(test_size=0.333, normalize=True, path="data"):
    mnist_train = datasets.MNIST(path, download=True, train=True)
    mnist_test = datasets.MNIST(path, download=True, train=False)

    x_train = mnist_train.data.numpy()
    y_train = mnist_train.targets.numpy()
    x_test = mnist_test.data.numpy()
    y_test = mnist_test.targets.numpy()

    if normalize:
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    x_test = x_test.reshape(-1, 784)
    x_train = x_train.reshape(-1, 784)

    return x_train, x_test, y_train, y_test


def load_aloi(test_size=0.333, normalize=True):
    X, y = fetch_libsvm("aloi")

    if normalize:
        # max value for aloi is 9
        X = X / 9.0

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size
    )

    return x_train, x_test, y_train, y_test


def load_rcv1(normalize=True):
    x_train, y_train = fetch_libsvm("rcv1.multiclass", normalize=normalize)
    x_test, y_test = fetch_libsvm("rcv1.multiclass_test", normalize=normalize)

    return x_train, x_test, y_train, y_test


def load_twentynews(normalize=True):
    x_train, y_train = fetch_20newsgroups_vectorized(
        subset="train", return_X_y=True, remove=("headers", "footers", "quotes")
    )
    x_test, y_test = fetch_20newsgroups_vectorized(
        subset="test", return_X_y=True, remove=("headers", "footers", "quotes")
    )

    if normalize:
        pass

    return x_train, x_test, y_train, y_test
