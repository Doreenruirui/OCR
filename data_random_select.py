from os.path import join, exists
import numpy as np
import os
import sys


folder_multi = '/scratch/dong.r/Dataset/OCR/multi'


def get_train_data(train_id, split_id, error_ratio, train_ratio, train):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    folder_error = join(folder_train, str(error_ratio))
    if not exists(folder_error):
        os.makedirs(folder_error)
    list_x = []
    for line in file(join(folder_error, train + '.x.txt')):
        list_x.append(line)
    list_y = []
    for line in file(join(folder_error, train + '.' + str(train_ratio) + '.y.txt')):
        list_y.append(line)
    list_info = []
    for line in file(join(folder_error, train + '.' + str(train_ratio) + '.info.txt')):
        list_info.append(line)
    num_line = len(list_x)
    rand_index = np.arange(num_line)
    np.random.shuffle(rand_index)
    num_train = int(np.floor(num_line * train_ratio * 0.01))
    train_index = rand_index[:num_train]
    f_x = open(join(folder_error, train + '.' + str(train_ratio) + '.x.txt'), 'w')
    f_y = open(join(folder_error, train + '.' + str(train_ratio) + '.y.txt'), 'w')
    f_info = open(join(folder_error, train + '.' + str(train_ratio) + '.info.txt'), 'w')
    for i in range(num_train):
        cur_index = train_index[i]
        f_x.write(list_x[cur_index])
        f_y.write(list_y[cur_index])
        f_info.write(list_info[cur_index])
    f_x.close()
    f_y.close()
    f_info.close()


arg_train_id = int(sys.argv[0])
arg_split_id = int(sys.argv[1])
arg_error_ratio = int(sys.argv[2])
arg_train_ratio = int(sys.argv[3])
arg_train = sys.argv[4]
get_train_data(arg_train_id, arg_split_id, arg_error_ratio, arg_train_ratio, arg_train)





