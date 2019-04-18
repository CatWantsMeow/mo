#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import wget
import tarfile


out_dir = 'data/svhn'

train_32_32 = ('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', 'train_32x32.mat')
test_32_32 = ('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', 'test_32x32.mat')
extra_32_32 = ('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat', 'extra_32x32.mat')

train_large = ('http://ufldl.stanford.edu/housenumbers/train.tar.gz', 'train.tar.gz')
test_large = ('http://ufldl.stanford.edu/housenumbers/test.tar.gz', 'test.tar.gz')
extra_large = ('http://ufldl.stanford.edu/housenumbers/extra.tar.gz', 'extra.tar.gz')


def download_data(url, filename, out_dir=out_dir):
    filename = os.path.join(out_dir, filename)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(filename):
        print(f"Downloading {filename}.")
        wget.download(url, filename)
        print()
    else:
        print(f"Skipping {filename} download (already exists)")


def extract_data(filename, out_dir=out_dir):
    filename = os.path.join(out_dir, filename)

    print(f"Extracting {filename}")
    with tarfile.open(filename) as tar:
        tar.extractall(out_dir)


if __name__ == '__main__':
    download_data(*train_32_32)
    download_data(*test_32_32)
    download_data(*extra_32_32)

    download_data(*train_large)
    download_data(*test_large)
    # download_data(*extra_large)

    extract_data(train_large[1])
    extract_data(test_large[1])
    # extract_data(extra_large[1])

