from os.path import join, exists
import numpy as np
import os
import sys
from collections import OrderedDict


folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR'

def get_train_info(cur_folder, train_ratio, train):
    folder_error = join(folder_multi, cur_folder)
    info = OrderedDict()
    for line in file(join(folder_error, train + '.info.txt')):
        info[line.split('\t')[0]] = 1
    num_info = len(info)
    rand_index = np.arange(num_info)
    np.random.shuffle(rand_index)
    num_train = int(np.floor(num_info * train_ratio * 0.01))
    train_index = np.sort(rand_index[:num_train])
    folder_out = join(folder_error, str(train_ratio))
    if not exists(folder_out):
        os.makedirs(folder_out)
    with open(join(folder_multi, cur_folder, str(train_ratio), 'index.txt'), 'w') as f_:
        info_dates = info.keys()
        for index in train_index:
            f_.write(info_dates[index] + '\n')
   
def get_train_data(cur_folder, train_ratio, train):
    folder_error = join(folder_multi, cur_folder)
    list_dates = []
    for line in file(join(folder_error, train + '.info.txt')):
        list_dates.append(line.split('\t')[0])
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
    train_dates = OrderedDict()
    for line in file(join(folder_error, str(train_ratio), 'index.txt')):
        train_dates[line.strip()] = 1
    train_index = []
    for i in range(len(list_dates)):
        if list_dates[i] in train_dates:
            train_index.append(i)
    folder_out = join(folder_error, str(train_ratio))
    f_x = open(join(folder_out, train + '.x.txt'), 'w')
    f_y = open(join(folder_out, train + '.y.txt'), 'w')
    if flag_info:
        f_info = open(join(folder_out, train + '.info.txt'), 'w')
    if flag_z:
        f_z = open(join(folder_out, train + '.z.txt'), 'w')
    #print len(list_z)
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
    np.savetxt(join(folder_out, train + '.index.txt'), train_index, fmt='%d')

arg_folder = sys.argv[1]
arg_train_ratio = float(sys.argv[2])
arg_train = sys.argv[3]
#get_train_info(arg_folder, arg_train_ratio, arg_train)
get_train_data(arg_folder, arg_train_ratio, arg_train)
