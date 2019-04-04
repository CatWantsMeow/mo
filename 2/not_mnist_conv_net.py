#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from time import time

import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils import load_not_mnist_data, remove_duplicates, flatten, to_one_hot


tf.logging.set_verbosity(tf.logging.ERROR)


class SimpleConvolutionalNN(object):

    def __init__(self, labels, img_train, labels_train, img_test, labels_test, **kwargs):
        self.labels = labels
        self.x_train = np.reshape(img_train, img_train.shape + (1,))
        self.y_train = to_one_hot(labels_train, len(labels))
        self.x_test = np.reshape(img_test, img_test.shape + (1,))
        self.y_test = to_one_hot(labels_test, len(labels))
        self.n = self.x_train.shape[1:]

        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(self.x_train, self.y_train, test_size=0.1)

        self.lr = kwargs.pop('learning_rate', 1e-3)
        self.history_path = kwargs.pop('results_path')
        self.model_path = kwargs.pop('model_path')

        self.init_nn()

    def init_nn(self):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Conv2D(16, 5, activation='relu', input_shape=self.n))
        self.model.add(keras.layers.Conv2D(16, 5, activation='relu'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(100, activation='relu'))
        self.model.add(keras.layers.Dropout(rate=0.1))
        self.model.add(keras.layers.Dense(self.y_train.shape[1], activation='softmax'))

        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print('Initialized basic net:')
        self.model.summary()

    def train(self, epochs=100, batch_size=1000):
        try:
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.x_val, self.y_val),
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10)
                ]
            )
        except KeyboardInterrupt:
            print()

        history = {k: [float(e) for e in v] for k, v in self.model.history.history.items()}
        with open(self.history_path, 'w+') as f:
            json.dump(history, f, indent=4)
            print(f'Saved history to {self.history_path}')

        self.model.save_weights(self.model_path)
        print(f'Saved model to {self.model_path}')

    def test(self):
        print(f'Evaluating accuracy of net on {len(self.x_test)} samples')

        history = {}
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                history = json.load(f)

        started = time()
        self.model.load_weights(self.model_path)
        _, acc = self.model.evaluate(self.x_test, self.y_test)
        print(f'Accuracy = {acc:.2f}')
        history['test_acc'] = float(acc)
        history['test_time'] = float(time() - started)

        with open(self.history_path, 'w+') as f:
            json.dump(history, f, indent=4)
            print(f'Saved history to {self.history_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['test', 'train'])
    parser.add_argument('net', type=str, choices=['basic'])
    args = parser.parse_args()

    labels, img_train, labels_train, img_test, labels_test = load_not_mnist_data()
    img_train, labels_train = remove_duplicates(img_train, labels_train, img_test)
    print()

    net = None
    if args.net == 'basic':
        net = SimpleConvolutionalNN(
            labels, img_train, labels_train, img_test, labels_test,
            model_path='models/not_mnist_conv_net_basic/model',
            results_path='results/not_mnist_conv_net_basic.json',
        )

    print()
    if net and args.action == 'train':
        net.train()
    elif net and args.action == 'test':
        net.test()