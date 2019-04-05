# -*- coding: utf-8 -*-
import numpy as np

from pathlib import Path
from scipy import io
from tensorflow import keras
from PIL import Image

from utils import to_one_hot


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    def to_x(a):
        x = np.array([np.array(Image.fromarray(i).resize((32, 32))) for i in a])
        return x.reshape(x.shape + (1,))

    def to_y(a):
        return to_one_hot(a, 10)

    x_train, y_train = to_x(x_train), to_y(y_train)
    x_test, y_test = to_x(x_test), to_y(y_test)
    print('Loaded and processes mnist dataset')
    return x_train, y_train, x_test, y_test


def load_single_digit_data(dir='data/svhn', extra=False, greyscale=True):
    def to_x(a):
        a = np.array([a[:,:,:,i] for i in range(a.shape[3])])
        if greyscale:
            return np.mean(a, axis=-1, keepdims=True).astype(np.uint8)
        return a

    def to_y(a):
        y = np.copy(a)
        y = y.reshape(y.shape[0])
        y[y == 10] = 0
        return to_one_hot(y, 10)

    def load_file(file):
        cache_file = Path(dir) / f"{file}.cache.npz"
        if cache_file.exists():
            f = np.load(cache_file)
            print(f'Loaded cached arrays for {file}')
            return [v for k, v in f.items()]

        f = io.loadmat(Path(dir) / file)
        x, y = to_x(f['X']), to_y(f['y'])
        np.savez(Path(dir) / f"{file}.cache.npz", x, y)
        print(f'Loaded and processed {file}')
        return x, y

    x_train, y_train = load_file('train_32x32.mat')
    x_test, y_test = load_file('test_32x32.mat')

    x_extra, y_extra = None, None
    if extra:
        x_extra, y_extra = load_file('extra_32x32.mat')

    return (
        x_train, y_train,
        x_test, y_test,
        x_extra, y_extra
    )



