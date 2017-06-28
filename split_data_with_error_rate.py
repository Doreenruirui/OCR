from os.path import join, exists
import numpy as np
import os
from collections import OrderedDict
from PyLib.operate_file import save_obj, load_obj
from levenshtein import align_pair
from multiprocessing import Pool

folder_data = '/scratch/dong.r/Dataset/OCR/data'
#folder_data = '/home/rui/Dataset/OCR/data/'


def split_data_with_error_rate(error_ratio, split_id):
    folder_out = join(folder_data, 'char_date_' + str(error_ratio) + '_' + str(split_id))
    with open(join(folder_out, 'train.x.txt'), 'r') as f_:
        list_x = f_.readlines()
    with open(join(folder_out, 'train.y.txt'), 'r') as f_:
        list_y = f_.readlines()
    P = Pool(40)
    dis = align_pair(P, list_x, list_y, 40, flag_char=1)
    error_rate = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    index = [[] for i in range(len(error_rate))]
    for i in range(len(list_x)):
        cur_dis = dis[i] * 1. / len(list_y[i])
        if cur_dis == 0:
            index[0].append(i)
        elif 0 < cur_dis < 0.05:
            index[1].append(i)
        elif 0.05 < cur_dis <= 0.1:
            index[2].append(i)
        elif 0.1 < cur_dis <= 0.15:
            index[3].append(i)
        elif 0.15 < cur_dis <= 0.2:
            index[4].append(i)
        elif 0.2 < cur_dis <= 0.25:
            index[5].append(i)
    print([len(ele) for ele in index])
    for i in range(len(error_rate)):
        folder_error = join(folder_out, 'error_' + str(error_rate[i]))
        if not exists(folder_error):
            os.makedirs(folder_error)
        file_error_x = join(folder_error, 'train.y.txt')
        with open(file_error_x, 'w') as f_:
            for j in index[i]:
                f_.write(list_x[j])
        file_error_y = join(folder_error, 'train.x.txt')
        with open(file_error_y, 'w') as f_:
            for j in index[i]:
                f_.write(list_y[j])

split_data_with_error_rate(25, 0)
