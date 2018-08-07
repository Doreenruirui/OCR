from os.path import join, exists
import numpy as np
import os
import sys


folder_root = '/gss_gpfs_scratch/dong.r/Dataset/OCR'


def get_random_index(folder_in, train):
    folder_data = join(folder_root, folder_in)
    num_line = 0
    for _ in file(join(folder_data, train + '.info.txt')):
        num_line += 1
    rand_index = np.arange(num_line)
    np.random.shuffle(rand_index)
    np.savetxt(join(folder_data, train + '.index.txt'), rand_index, fmt='%d')


def get_train_data(folder_in, train_ratio, prefix, list_postfix):
    folder_data = join(folder_root, folder_in)
    rand_index = np.loadtxt(join(folder_data, prefix + '.index.txt'), dtype=int)
    num_line = len(rand_index)
    num_train = int(np.floor(num_line * train_ratio * 0.01))
    train_index = rand_index[:num_train]
    folder_out = join(folder_data, str(train_ratio))
    if not exists(folder_out):
        os.makedirs(folder_out)
    for postfix in list_postfix:
        cur_list = []
        for line in file(join(folder_data,  prefix + '.' + postfix + '.txt')):
            cur_list.append(line)
        with open(join(folder_out, prefix + '.' + postfix + '.txt'), 'w') as f_:
            for i in range(num_train):
                cur_index = train_index[i]
                f_.write(cur_list[cur_index])


task = int(sys.argv[1])
arg_folder = sys.argv[2]
arg_train = sys.argv[3]
if task == 0:
    get_random_index(arg_folder, arg_train)
else:
    arg_train_ratio = float(sys.argv[4])
    arg_postfix = sys.argv[5].split('_')
    get_train_data(arg_folder, arg_train_ratio, arg_train, arg_postfix)
