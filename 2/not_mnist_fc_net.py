#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import numpy as np
import tensorflow as tf

from time import time
from sklearn.model_selection import train_test_split

from utils import flatten, to_one_hot
from not_mnist_preprocess import load_not_mnist_data, remove_duplicates


tf.logging.set_verbosity(tf.logging.ERROR)


class FullyConnectedNet(object):

    def __init__(self, labels, img_train, labels_train, img_test, labels_test, **kwargs):
        self.labels = labels
        self.x_train = flatten(img_train)
        self.y_train = to_one_hot(labels_train, len(labels))
        self.x_test = flatten(img_test)
        self.y_test = to_one_hot(labels_test, len(labels))

        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(self.x_train, self.y_train, test_size=0.1)

        self.n = self.x_train.shape[1]
        self.m = self.y_train.shape[1]
        self.k = kwargs.pop('hidden_layer_size', 256)
        self.lr = kwargs.pop('learning_rate', 1e-3)

        self.dropout = kwargs.pop('dropout', False)
        self.regularization = kwargs.pop('regularization', False)
        self.adaptive_lr = kwargs.pop('adaptive_lr', False)

        self.history_path = kwargs.pop('results_path')
        self.model_path = kwargs.pop('model_path')

        self.init_nn()
        self.saver = tf.train.Saver()

    def init_nn(self):
        self.input = tf.placeholder(tf.float32, [None, self.x_train.shape[1]], name='inputs')
        self.expected = tf.placeholder(tf.float32, [None, self.y_train.shape[1]], name='expected')
        self.dropout_rate = tf.placeholder(tf.float32)

        self.w = {}
        self.b = {}

        self.w[0] = tf.Variable(tf.truncated_normal((self.n, self.k), stddev=0.05))
        self.b[0] = tf.Variable(tf.zeros(self.k))
        hidden = tf.nn.relu(tf.matmul(self.input, self.w[0]) + self.b[0])

        if self.dropout:
            hidden = tf.nn.dropout(hidden, rate=self.dropout_rate)

        self.w[1] = tf.Variable(tf.truncated_normal((self.k, self.k), stddev=0.05))
        self.b[1] = tf.Variable(tf.zeros(self.k))
        hidden = tf.nn.relu(tf.matmul(hidden, self.w[1]) + self.b[1])

        if self.dropout:
            hidden = tf.nn.dropout(hidden, rate=self.dropout_rate)

        self.w[2] = tf.Variable(tf.truncated_normal((self.k, self.m), stddev=0.05))
        self.b[2] = tf.Variable(tf.zeros(self.m))
        self.logits = tf.matmul(hidden, self.w[2]) + self.b[2]
        self.predicted = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits,
                labels=self.expected
            )
        )
        if self.regularization:
            regularizers = tf.nn.l2_loss(self.w[0]) + \
                           tf.nn.l2_loss(self.w[1]) + \
                           tf.nn.l2_loss(self.w[2])
            self.loss = tf.reduce_mean(self.loss + 0.005 * regularizers)

        if self.adaptive_lr:
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(self.predicted, 1), tf.argmax(self.expected, 1)),
                tf.float32,
            )
        )

        print(f'Initialized net (dropout: {self.dropout}, '
              f'regularization : {self.regularization}, '
              f'adaptive_lr : {self.adaptive_lr})')

    def train(self, epochs=500, batch_size=1000):
        print(f'Training net on {len(self.x_train)} samples, '
              f'validating on {len(self.x_val)} samples')

        history = {}
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                history = json.load(f)

        indices = np.arange(self.x_train.shape[0])
        with tf.Session() as session:
            session.run(tf.local_variables_initializer())
            session.run(tf.global_variables_initializer())

            try:
                history = {
                    "loss": [],
                    "categorical_accuracy": [],
                    "val_loss": [],
                    "val_categorical_accuracy": []
                }

                best_loss = 1e10
                no_improvement = 0
                started = time()
                for epoch in range(epochs):
                    print(f"Epoch #{epoch:<4} [", end='')

                    np.random.shuffle(indices)
                    batches = np.arange(0, indices.shape[0], batch_size)

                    train_loss = 0
                    train_acc = 0

                    z = self.x_train.shape[0] // batch_size // 20 + 1
                    for i, j in enumerate(batches):
                        _, loss, acc = session.run(
                            (self.optimizer, self.loss, self.accuracy),
                            feed_dict={
                                self.input: self.x_train[indices[j:j + batch_size]],
                                self.expected: self.y_train[indices[j:j + batch_size]],
                                self.dropout_rate: 0.1,
                            }
                        )

                        train_loss += loss / len(batches)
                        train_acc += acc / len(batches)

                        if i % z == 0:
                            print('.', end='', flush=True)

                    print('] ', end='', flush=True)

                    feed_dict = {
                        self.input: self.x_val,
                        self.expected: self.y_val,
                        self.dropout_rate: 0
                    }
                    val_loss = self.loss.eval(feed_dict=feed_dict)
                    val_acc = self.accuracy.eval(feed_dict=feed_dict)

                    print(f"loss = {train_loss:<10.6f}"
                          f"acc = {train_acc:<10.6f}"
                          f"val_loss = {val_loss:<10.6f}"
                          f"val_acc = {val_acc:<10.6f}", end='')

                    history['loss'].append(float(train_loss))
                    history['val_loss'].append(float(val_loss))
                    history['categorical_accuracy'].append(float(train_acc))
                    history['val_categorical_accuracy'].append(float(val_acc))

                    if best_loss > val_loss:
                        no_improvement = 0
                        best_loss = val_loss
                        self.saver.save(session, self.model_path)
                        print('model saved.', end='')
                    print()

                    no_improvement += 1
                    if no_improvement > 20:
                        break

            except KeyboardInterrupt:
                print()

            history['train_time'] = float(time() - started)
            with open(self.history_path, 'w+') as f:
                json.dump(history, f, indent=4)
                print(f'Saved history to {self.model_path}')

    def test(self):
        print(f'Evaluating accuracy of net on {len(self.x_test)} samples')

        results = {}
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                results = json.load(f)

        with tf.Session() as session:
            self.saver.restore(session, self.model_path)

            started = time()
            acc = self.accuracy.eval({
                self.input: self.x_test,
                self.expected: self.y_test,
                self.dropout_rate: 0
            })
            print(f'Accuracy = {acc:.5f}')
            results['test_acc'] = float(acc)
            results['test_time'] = float(time() - started)

        with open(self.history_path, 'w+') as f:
            json.dump(results, f, indent=4)

    def get_wrong_predictions(self):
        print(f'Evaluating of net on {len(self.x_test)} samples')

        with tf.Session() as session:
            self.saver.restore(session, self.model_path)
            return self.predicted.eval({
                self.input: self.x_test,
                self.expected: self.y_test,
                self.dropout_rate: 0
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['test', 'train'])
    parser.add_argument('net', type=str, choices=['basic', 'dropout', 'adaptive_lr'])
    args = parser.parse_args()

    labels, img_train, labels_train, img_test, labels_test = load_not_mnist_data()
    img_train, labels_train = remove_duplicates(img_train, labels_train, img_test)

    net = None
    if args.net == 'basic':
        net = FullyConnectedNet(
            labels, img_train, labels_train, img_test, labels_test,
            model_path='models/not_mnist_fc_net_basic/model.ckpt',
            results_path='results/not_mnist_fc_net_basic.json',
        )

    elif args.net == 'dropout':
        net = FullyConnectedNet(
            labels, img_train, labels_train, img_test, labels_test,
            model_path='models/not_mnist_fc_net_dropout/model.ckpt',
            results_path='results/not_mnist_fc_net_dropout.json',
            dropout=True,
            regularization=True
        )

    if args.net == 'adaptive_lr':
        net = FullyConnectedNet(
            labels, img_train, labels_train, img_test, labels_test,
            model_path='models/not_mnist_fc_net_adaptive_lr/model.ckpt',
            results_path='results/not_mnist_fc_net_adaptive_lr.json',
            learning_rate=1e-3,
            dropout=True,
            regularization=False,
            adaptive_lr=True
        )

    if net and args.action == 'train':
        net.train()
    elif net and args.action == 'test':
        net.test()
