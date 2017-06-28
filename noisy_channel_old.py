from os.path import join, exists
import numpy as np
import os
from collections import OrderedDict
from PyLib.operate_file import save_obj, load_obj
from levenshtein import recover_pair
from multiprocessing import Pool

folder_data = '/scratch/dong.r/Dataset/OCR/data'
#folder_data = '/home/rui/Dataset/OCR/data/'


def generate_file_for_align(error_rate):
    folder_out = join(folder_data, 'char_' + str(error_ratio))
    with open(join(folder_out, 'train.x.txt'), 'r') as f_:
        list_x = f_.readlines()
    with open(join(folder_out, 'train.y.txt'), 'r') as f_:
        list_y = f_.readlines()
    with open(join(folder_out, 'train_data.txt'), 'w') as f_:
        for i in range(len(list_x)):
            x = []
            for ele in list_x[i].strip():
                if ele == '_':
                    ele = 'UNDERLINE'
                elif ele == ' ':
                    ele = '_'
                x.append(ele)
            y = []
            for ele in list_y[i].strip():
                if ele == '_':
                    ele = 'UNDERLINE'
                elif ele == ' ':
                    ele = '_'
                y.append(ele)
            f_.write(' '.join(y)
                     + '\t' + ' '.join(x) + '\n')


def generate_noisy_channel(error_ratio, split_id, nthread):
    folder_out = join(folder_data, 'char_' + str(error_ratio))
    with open(join(folder_out, 'train.x.txt'), 'r') as f_:
        list_x = f_.readlines()
    with open(join(folder_out, 'train.y.txt'), 'r') as f_:
        list_y = f_.readlines()

    P = Pool(nthread)
    res_op = recover_pair(P, list_y, list_x, nthread)
    # res_op = {}
    # for i in range(len(list_x)):
    #     cur_op = recover(list_y[i].strip(), list_x[i].strip())
    #     for ele in cur_op:
    #         if ele not in res_op:
    #             res_op[ele] = {}
    #         for k in cur_op[ele]:
    #             res_op[ele][k] = res_op[ele].get(k, 0) + cur_op[ele][k]
    save_obj(join(folder_out, 'levenshtein'), res_op)

#generate_file_for_align(25, 0)
generate_noisy_channel(25, 0, 50)
