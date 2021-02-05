import torch
import numpy as np
import math
import random
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_minibatches_idx(n, minibatch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(math.ceil(n / minibatch_size)):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return minibatches


def get_weighted_minibatches_idx(n, minibatch_size, weighted_idx, ratio, shuffle=True):
    idx_list = np.arange(n, dtype="int32")
    extra_idx = np.array(weighted_idx * (ratio - 1), dtype='int32')
    idx_list = np.concatenate((idx_list, extra_idx))
    if shuffle:
        np.random.shuffle(idx_list)
    n = len(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(math.ceil(n / minibatch_size)):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return minibatches


