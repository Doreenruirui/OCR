import numpy as np


def split_train_test_with_ratio(data, ratio):
    num_data = len(data)
    num_train = int(np.floor(num_data * ratio))
    rand_index = np.arange(num_data)
    np.random.shuffle(rand_index)
    index_train = rand_index[:num_train]
    index_test = rand_index[num_train:]
    return index_train, index_test


