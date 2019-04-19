#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pickle
import json

from time import time
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split

from utils import flatten
from not_mnist_preprocess import load_not_mnist_data, remove_duplicates


model_path = 'models/not_mnist_logistic_regression.model'
results_path = 'results/not_mnist_logistic_regression.json'


def load_data():
    labels, img_train, labels_train, img_test, labels_test = load_not_mnist_data()
    img_train, labels_train = remove_duplicates(img_train, labels_train, img_test)
    return labels, flatten(img_train), labels_train, flatten(img_test), labels_test


def train():
    _, x_train, y_train, x_test, y_test = load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)

    results.setdefault('val_acc', {})
    ns = [int(2 ** n) for n in np.arange(7, np.ceil(np.log2(len(x_train))) + 1)]

    print('Training Logistic Regression model:')
    for n in ns:
        indices = np.arange(len(x_train))
        if n < len(x_train):
            indices = np.random.choice(indices, n, replace=False)

        # model = SGDClassifier(loss='log', tol=1e-4, early_stopping=True)
        model = LogisticRegression()
        model.fit(x_train[indices], y_train[indices])

        y_pred = model.predict(x_val)
        results["val_acc"][str(n)] = acc = np.mean(np.equal(y_pred, y_val).astype(np.int))
        print(f"n = {n}, accuracy = {acc:.5f}, iterations = {model.n_iter_}")

        with open(results_path, 'w+') as f:
            json.dump(results, f, indent=4)

        with open(model_path, 'wb+') as f:
            pickle.dump(model, f)


def test():
    if not os.path.exists(model_path):
        print('Model file does not exist')
        return

    _, _, _, x_test, y_test = load_data()

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)

    started = time()
    y_pred = model.predict(x_test)
    acc = np.mean(np.equal(y_pred, y_test).astype(np.int))
    print(f"Accuracy = {acc:.5f}")

    results["test_acc"] = acc
    results["test_time"] = time() - started
    with open(results_path, 'w+') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['test', 'train'])

    args = parser.parse_args()
    if args.action == 'test':
        test()
    elif args.action == 'train':
        train()
