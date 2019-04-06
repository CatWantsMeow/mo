# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import h5py

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
    print('Loaded and processed mnist dataset')
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


def load_multiple_digits_data(dir='data/svhn', train=True, extra=False):

    def parse_digit_struct(file):
        if Path(f"{file}.cache.json").exists():
            with open(f"{file}.cache.json", "r") as f:
                images = json.load(f)
                print(f'Loaded cached image attrs from {file}.cache.json')
                return images

        f = h5py.File(file, 'r')
        print(f'Opened file {file}')

        names = f['digitStruct']['name']
        bbox = f['digitStruct']['bbox']

        def extract_name(i):
            return ''.join([chr(c[0]) for c in f[names[i][0]].value])

        def extract_attr(i, attr):
            attr = f[bbox[i].item()][attr]
            if len(attr) > 1:
                return [f[attr.value[j].item()].value[0][0] for j in range(len(attr))]
            else:
                return [attr.value[0][0]]

        images = {}
        print(f'Extracting image attrs from {file}: ', end='')
        for i in range(len(names)):
            name = extract_name(i)
            images[name] = {
                "label": extract_attr(i, 'label'),
                "top": extract_attr(i, 'top'),
                "left": extract_attr(i, 'left'),
                "height": extract_attr(i, 'height'),
                "width": extract_attr(i, 'width')
            }
            if i % 1000 == 0:
                print('.', end='', flush=True)
        print()

        with open(f"{file}.cache.json", 'w+') as f:
            json.dump(images, f)
        return images

    def process_images(dir):
        cache_file = Path(dir) / 'cache.npz'
        if cache_file.exists():
            f = np.load(cache_file)
            print(f'Loaded cached arrays for {dir}')
            return [v for k, v in f.items()]

        attrs = parse_digit_struct(Path(dir) / 'digitStruct.mat')

        x, y = [], []
        print(f'Processing images from {dir}: ', end='', flush=True)
        for i, name in enumerate(os.listdir(dir)):
            if name not in attrs:
                print('s', end='', flush=True)
                continue

            img = Image.open(Path(dir) / name)

            height = int(max(attrs[name]['height']))
            width = int(max(attrs[name]['width']))
            left = max(int(min(attrs[name]['left'])) - 0.5 * width, 0)
            top = max(int(min(attrs[name]['top'])) - 0.5 * height, 0)
            right = min(int(max(attrs[name]['left'])) + 1.5 * width, img.size[0])
            bottom = min(int(max(attrs[name]['top'])) + 1.5 * height, img.size[1])

            img = img.crop(box=(left, top, right, bottom))
            img = img.resize((96, 96))

            label = [d % 10 for d in attrs[name]['label']]
            if len(label) > 6:
                print('e', end='', flush=True)
                continue

            label += [10] * (6 - len(label))
            label = to_one_hot(np.array(label, dtype=np.int), 11)

            x.append(np.array(img))
            y.append(np.array(label))

            if i % 1000 == 0:
                print('.', end='', flush=True)
        print()

        x = np.array(x, dtype=np.uint8)
        y = np.array(y, dtype=np.uint8)
        np.savez(Path(dir) / "cache.npz", x, y)
        return x, y

    x_test, y_test = process_images(Path(dir) / 'test/')

    x_train, y_train = None, None
    if train:
        x_train, y_train = process_images(Path(dir) / 'train/')

    x_extra, y_extra = None, None
    if extra:
        x_extra, y_extra = process_images(Path(dir) / 'extra/')

    return (
        x_train, y_train,
        x_test, y_test,
        x_extra, y_extra
    )


if __name__ == '__main__':
    load_multiple_digits_data()
