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

from svhn_preprocess import load_multiple_digits_data


tf.logging.set_verbosity(tf.logging.ERROR)


def to_y(a, n):
    return [a[:,i,:] for i in range(n)]


class MobileNet(object):

    def __init__(self, x_train, y_train, x_test, y_test, **kwargs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(self.x_train, self.y_train, test_size=0.1)

        self.y_train = to_y(self.y_train, 6)
        self.y_val = to_y(self.y_val, 6)
        self.y_test = to_y(self.y_test, 6)

        self.lr = kwargs.pop('learning_rate', 1e-3)
        self.history_path = kwargs.pop('results_path')
        self.model_path = kwargs.pop('model_path')

        self.init_nn()

    def init_nn(self):
        input = keras.layers.Input(shape=(96, 96, 3))

        mobile_net = keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(96, 96, 3),
            input_tensor=input,
            pooling='avg'
        )

        outputs = [
            keras.layers.Dense(11, activation='softmax', name=f'out_{i}')(mobile_net.output)
            for i in range(6)
        ]

        self.model = keras.models.Model(
            inputs=[input],
            outputs=outputs
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=self.lr),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'],
            loss_weights=[1, 1, 0.5, 0.3, 0.1, 0.05]
        )

        print('Initialized mobile net')

    def train(self, epochs=100, batch_size=32):
        started = time()
        try:
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=epochs,
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
        y_pred = self.model.predict(self.x_test)

        total = np.array([True] * len(self.x_test))
        for i, (y1, y2) in enumerate(zip(self.y_test, y_pred)):
            cur = np.argmax(y1, axis=1) == np.argmax(y2, axis=1)
            total = np.logical_and(total, cur)

            acc = np.mean(cur.astype(np.int))
            history[f'test_acc_out_{i}'] = acc
            print(f'Accuracy of out_{i} = {acc:.5f}')

        acc = np.mean(total.astype(np.int))
        history['test_acc'] = acc
        print(f'Accuracy = {acc:.5f}')

        history['test_time'] = float(time() - started)

        with open(self.history_path, 'w+') as f:
            json.dump(history, f, indent=4)
            print(f'Saved history to {self.history_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['test', 'train'])
    parser.add_argument('data', type=str, choices=['basic', 'extra'])
    args = parser.parse_args()

    net = None
    if args.data == 'basic':
        x_train, y_train, x_test, y_test, _, _ = load_multiple_digits_data()
        net = MobileNet(
            x_train, y_train, x_test, y_test,
            model_path='models/svhn_multiple_mobile_net_basic/model',
            results_path='results/svhn_multiple_mobile_net_basic.json',
        )

    elif args.data == 'extra':
        if args.action == 'train':
            _, _, x_test, y_test, x_extra, y_extra = load_multiple_digits_data(extra=True)
            net = MobileNet(
                x_extra, y_extra, x_test, y_test,
                model_path='models/svhn_multiple_mobile_net_extra/model',
                results_path='results/svhn_multiple_mobile_net_extra.json',
            )
            net.model.load_weights('models/svhn_multiple_mobile_net_basic/model')

        else:
            x_train, y_train, x_test, y_test, _, _ = load_multiple_digits_data(extra=False)
            net = MobileNet(
                x_train, y_train, x_test, y_test,
                model_path='models/svhn_multiple_mobile_net_extra/model',
                results_path='results/svhn_multiple_mobile_net_extra.json',
            )

    if net and args.action == 'train':
        net.train()
    elif net and args.action == 'test':
        net.test()
