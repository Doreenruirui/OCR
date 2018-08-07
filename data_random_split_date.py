from os.path import join, exists
import numpy as np
import os
import sys
from collections import OrderedDict


folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR'


def get_train_info(cur_folder, train):
    folder_data = join(folder_multi, cur_folder)
    info = OrderedDict()
    for line in file(join(folder_data, train + '.info.txt')):
        info[line.split('\t')[0]] = 1
    num_info = len(info)
    rand_index = np.arange(num_info)
    np.random.shuffle(rand_index)
    with open(join(folder_data, train + '.info.random.txt'), 'w') as f_:
        all_info = info.keys()
        for index in rand_index:
            f_.write(all_info[index] + '\n')


def get_train_data(cur_folder, num_split, train, split_id):
    folder_data = join(folder_multi, cur_folder)
    list_dates = []
    for line in file(join(folder_data, train + '.info.txt')):
        list_dates.append(line.split('\t')[0])
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
    all_info = []
    for line in file(join(folder_data, train + '.info.random.txt')):
        all_info.append(line.strip())
    num_info = len(all_info)
    size_split = int(np.ceil(num_info * 1. / num_split))
    num_train = size_split * split_id
    folder_out = join(folder_data, str(int(np.floor(100. * split_id / num_split))))
    if not exists(folder_out):
        os.makedirs(folder_out)
    train_dates = OrderedDict()
    for i in range(num_train):
        train_dates[all_info[i]] = 1
    train_index = []
    for i in range(len(list_dates)):
        if list_dates[i] in train_dates:
            train_index.append(i)
    f_x = open(join(folder_out, train + '.x.txt'), 'w')
    f_y = open(join(folder_out, train + '.y.txt'), 'w')
    if flag_info:
        f_info = open(join(folder_out, train + '.info.txt'), 'w')
    if flag_z:
        f_z = open(join(folder_out, train + '.z.txt'), 'w')
    for i in range(len(train_index)):
        cur_index = train_index[i]
        print cur_index
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
    get_train_info(arg_folder, arg_train)
else:
    arg_num_split = int(sys.argv[4])
    arg_split_id = int(sys.argv[5])
    get_train_data(arg_folder, arg_num_split, arg_train, arg_split_id)
