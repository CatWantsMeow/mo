# -*- coding: utf-8 -*-
import numpy as np


def flatten(a):
    return a.reshape(a.shape[0], a.shape[1] * a.shape[2])


def to_one_hot(a, n):
    result = np.zeros(shape=(a.shape[0], n))
    result[np.arange(len(a)), a] = 1
    return result
