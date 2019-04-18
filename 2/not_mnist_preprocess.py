# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
from PIL import Image


def load_images(path, n):
    labels = ['I', 'G', 'A', 'F', 'H', 'J', 'C', 'D', 'E', 'B']

    x, y = [], []
    for i, l in enumerate(labels):
        d = Path(path) / l
        print(f'Loading {str(d)} ', end='')
        for j, f in zip(range(n), d.iterdir()):
            try:
                with Image.open(f) as img:
                    x.append(np.array(img))
                    y.append(i)
            except OSError:
                pass
            if j % 1000 == 0:
                print('.', end='', flush=True)
        print(flush=True)
    return np.array(labels), np.array(x), np.array(y)


def remove_duplicates(img_train, labels_train, img_test):
    img_new, labels_new = [], []
    test_set = {e.tostring() for e in img_test}
    for i, (x, y) in enumerate(zip(img_train, labels_train)):
        if x.tostring() not in test_set:
            img_new.append(x)
            labels_new.append(y)

    print(f'Removed {img_train.shape[0] - len(img_new)} duplicated images')
    return np.array(img_new), np.array(labels_new)


def load_not_mnist_data(path='data/not_mnist/', use_cache=True):
    train_folder = Path(path) / 'notMNIST_large'
    test_folder = Path(path) / 'notMNIST_small'

    train_cache_file = Path(path) / 'train.npz'
    test_cache_file = Path(path) / 'test.npz'

    if train_cache_file.exists() and test_cache_file.exists() and use_cache:
        f = np.load(train_cache_file)
        labels, img_train, labels_train = [v for k, v in f.items()]
        f = np.load(test_cache_file)
        labels, img_test, labels_test = [v for k, v in f.items()]
        print('Loaded cached arrays')

    else:
        labels, img_train, labels_train = load_images(train_folder, 10000000)
        labels, img_test, labels_test = load_images(test_folder, 10000000)
        np.savez(train_cache_file, labels, img_train, labels_train)
        np.savez(test_cache_file, labels, img_test, labels_test)

    return labels, img_train, labels_train, img_test, labels_test


if __name__ == '__main__':
    load_not_mnist_data()
