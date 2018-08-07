from os.path import join, exists
import numpy as np
import os
import sys


folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR'


def shuffle_index(cur_folder, train):
    folder_data = join(folder_multi, cur_folder)
    list_info = []
    for line in file(join(folder_data, train + '.info.txt')):
        list_info.append(line)
    num_line = len(list_info)
    rand_index = np.arange(num_line)
    np.random.shuffle(rand_index)
    np.savetxt(join(folder_data,  train + '.index.txt'), rand_index, fmt='%d')


def get_train_data(cur_folder, num_split, train, split_id):
    folder_data = join(folder_multi, cur_folder)
    list_x = []
    for line in file(join(folder_data, train + '.x.txt')):
        list_x.append(line)
    list_y = []
    for line in file(join(folder_data, train + '.y.txt')):
        list_y.append(line)
    flag_z = 0
    if os.path.exists(join(folder_data, train + '.z.txt')):
        list_z = []
        flag_z = 1
        for line in file(join(folder_data, train + '.z.txt')):
            list_z.append(line)
    flag_info = 0
    if os.path.exists(join(folder_data, train + '.info.txt')):
        list_info = []
        flag_info = 1
        for line in file(join(folder_data, train + '.info.txt')):
            list_info.append(line)
    num_line = len(list_x)
    rand_index = np.loadtxt(join(folder_data, train + '.index.txt'), dtype=int)
    size_split = int(np.ceil(num_line * 1. / num_split))
    num_train = split_id * size_split
    train_index = rand_index[: num_train]
    folder_out = join(folder_data,  str(int(np.floor(100. * split_id / num_split))))
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

arg_shuffle = int(sys.argv[1])
arg_folder = sys.argv[2]
arg_train = sys.argv[3]
if arg_shuffle:
    shuffle_index(arg_folder, arg_train)
else:
    arg_num_split = int(sys.argv[4])
    arg_split_id = int(sys.argv[5])
    get_train_data(arg_folder, arg_num_split, arg_train, arg_split_id)
