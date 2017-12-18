from os.path import join, exists
import numpy as np
import os
import sys


folder_multi = '/scratch/dong.r/Dataset/OCR'


def get_train_data(cur_folder, train_ratio, train):
    folder_error = join(folder_multi, cur_folder)
    list_x = []
    for line in file(join(folder_error, train + '.x.txt')):
        list_x.append(line)
    list_y = []
    for line in file(join(folder_error, train + '.y.txt')):
        list_y.append(line)
    flag_z = 0
    if os.path.exists(join(folder_error, train + '.z.txt')):
        list_z = []
        flag_z = 1
        for line in file(join(folder_error, train + '.z.txt')):
            list_z.append(line)
    flag_info = 0
    if os.path.exists(join(folder_error, train + '.info.txt')):
        list_info = []
        flag_info = 1
        for line in file(join(folder_error, train + '.info.txt')):
            list_info.append(line)
    num_line = len(list_x)
    rand_index = np.arange(num_line)
    np.random.shuffle(rand_index)
    num_train = int(np.floor(num_line * train_ratio * 0.01))
    train_index = rand_index[:num_train]
    folder_out = join(folder_error, str(train_ratio))
    if not exists(folder_out):
        os.makedirs(folder_out) 
    f_x = open(join(folder_out, train + '.x.txt'), 'w')
    f_y = open(join(folder_out, train + '.y.txt'), 'w')
    if flag_info:
        f_info = open(join(folder_out, train + '.info.txt'), 'w')
    if flag_z:
        f_z = open(join(folder_out, train + '.z.txt'), 'w')
    for i in range(num_train):
        cur_index = train_index[i]
        f_x.write(list_x[cur_index])
        f_y.write(list_y[cur_index])
        if flag_z:
            f_z.write(list_z[cur_index])
        if flag_info:
            f_info.write(list_info[cur_index])
    f_x.close()
    f_y.close()
    if flag_z:
        f_z.close()
    if flag_info:
        f_info.close()
    np.savetxt(join(folder_out, train + '.index.txt'), train_index, fmt='%d')

arg_folder = sys.argv[1]
arg_train_ratio = float(sys.argv[2])
arg_train = sys.argv[3]
get_train_data(arg_folder, arg_train_ratio, arg_train)
