#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import numpy as np
import tensorflow as tf

from time import time
from tensorflow import keras
from sklearn.model_selection import train_test_split

from svhn_preprocess import load_mnist, load_single_digit_data


tf.logging.set_verbosity(tf.logging.ERROR)


class ConvNet(object):

    def __init__(self, x_train, y_train, x_test, y_test, **kwargs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n = self.x_train.shape[1:]

        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(self.x_train, self.y_train, test_size=0.1)

        self.lr = kwargs.pop('learning_rate', 1e-3)
        self.history_path = kwargs.pop('results_path')
        self.model_path = kwargs.pop('model_path')

        self.init_nn()

    def init_nn(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(16, 5, activation='relu', input_shape=self.n, padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), padding = 'same'),
            keras.layers.Conv2D(32, 5, activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
            keras.layers.Conv2D(64, 5, activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Dense(self.y_train.shape[1], activation='softmax')
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=self.lr),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )

        print('Initialized basic net:')
        self.model.summary()

        # input = keras.layers.Input(shape=self.n)
        # mobile_net = keras.applications.mobilenet_v2.MobileNetV2(
        #     include_top=False,
        #     weights='imagenet',
        #     input_shape=self.n,
        #     input_tensor=input,
        #     pooling='avg'
        # )
        # dropout = keras.layers.Dropout(rate=0.1)(mobile_net.output)
        # output = keras.layers.Dense(11, activation='softmax')
        #
        # self.model = keras.models.Model(
        #     inputs=[input],
        #     outputs=[output]
        # )
        # self.model.compile(
        #     optimizer=keras.optimizers.Adam(lr=self.lr),
        #     loss='categorical_crossentropy',
        #     metrics=['categorical_accuracy']
        # )
        #
        # print('Initialized mobile net')

    def train(self, epochs=100, batch_size=100):
        started = time()
        try:
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=epochs,
                verbose=2,
                batch_size=batch_size,
                validation_data=(self.x_val, self.y_val),
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
        except KeyboardInterrupt:
            print()

        history = {k: [float(e) for e in v] for k, v in self.model.history.history.items()}
        history['train_time'] = float(time() - started)
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
        print(f'Accuracy = {acc:.5f}')
        history['test_acc'] = float(acc)
        history['test_time'] = float(time() - started)

        with open(self.history_path, 'w+') as f:
            json.dump(history, f, indent=4)
            print(f'Saved history to {self.history_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['test', 'train'])
    parser.add_argument('data', type=str, choices=['mnist', 'svhn', 'svhn_extra'])
    args = parser.parse_args()

    net = None
    if args.data == 'mnist':
        x_train, y_train, _, _ = load_mnist()
        _, _, x_test, y_test, _, _ = load_single_digit_data(extra=False)
        net = ConvNet(
            x_train, y_train, x_test, y_test,
            model_path='models/svhn_mnist_conv_net_mnist/model',
            results_path='results/svhn_mnist_conv_net_mnist.json',
        )

    elif args.data == 'svhn':
        x_train, y_train, x_test, y_test, _, _ = load_single_digit_data(extra=False)
        net = ConvNet(
            x_train, y_train, x_test, y_test,
            model_path='models/svhn_mnist_conv_net_svhn/model',
            results_path='results/svhn_mnist_conv_net_svhn.json',
        )

    elif args.data == 'svhn_extra':
        if args.action == 'train':
            _, _, x_test, y_test, x_extra, y_extra = load_single_digit_data(extra=True)
            net = ConvNet(
                x_extra, y_extra, x_test, y_test,
                model_path='models/svhn_mnist_conv_net_svhn_extra/model',
                results_path='results/svhn_mnist_conv_net_svhn_extra.json',
            )
            net.model.load_weights('models/svhn_mnist_conv_net_svhn/model')

        else:
            x_train, y_train, x_test, y_test, _, _ = load_single_digit_data(extra=False)
            net = ConvNet(
                x_train, y_train, x_test, y_test,
                model_path='models/svhn_mnist_conv_net_svhn_extra/model',
                results_path='results/svhn_mnist_conv_net_svhn_extra.json',
            )

    print()
    if net and args.action == 'train':
        net.train()
    elif net and args.action == 'test':
        net.test()
