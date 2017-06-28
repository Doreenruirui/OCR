from os.path import join, exists
import numpy as np
import os
from collections import OrderedDict
from PyLib.operate_file import save_obj, load_obj
from levenshtein import recover_pair, recover_thread
from multiprocessing import Pool
import re

folder_data = '/scratch/dong.r/Dataset/OCR/data'
#folder_data = '/home/rui/Dataset/OCR/data/'


def generate_file_for_align(error_ratio, split_id):
    folder_out = join(folder_data, 'char_date_' + str(error_ratio) + '_' + str(split_id))
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
    def remove(text):
        return re.sub(r'[^\x00-\x7F]', '', text)
    folder_out = join(folder_data, 'char_date_' + str(error_ratio) + '_' + str(split_id), 'cmp')
    with open(join(folder_out, 'x.txt'), 'r') as f_:
        list_x = f_.readlines()
        list_x = [remove(ele.strip()) for ele in list_x]
    with open(join(folder_out, 'y.txt'), 'r') as f_:
        list_y = f_.readlines()
        list_y = [remove(ele.strip()) for ele in list_y]
    _, res_op, begin, middle, end = recover_thread(list_y, list_x, 0)
    #P = Pool(nthread)
    #res_op, begin, middle, end = recover_pair(P, list_y, list_x, nthread)
    #save_obj(join(folder_out, 'levenshtein'), res_op)
    #print begin, middle, end


def analyze_noisy_channel(error_ratio, split_id):
    folder_out = join(folder_data, 'char_date_' + str(error_ratio) + '_' + str(split_id))
    res_op1 = load_obj(join(folder_out, 'cmp', 'levenshtein1'))
    res_op2 = load_obj(join(folder_out, 'cmp', 'levenshtein2'))
    dict_char = {}
    char_id = 0
    for char1 in res_op1:
        if char1 not in dict_char:
            dict_char[char1] = char_id
            char_id += 1
        for char2 in res_op1[char1]:
            if char2 not in dict_char:
                dict_char[char2] = char_id
                char_id += 1
    for char1 in res_op2:
        if char1 not in dict_char:
            dict_char[char1] = char_id
            char_id += 1
        for char2 in res_op2[char1]:
            if char2 not in dict_char:
                dict_char[char2] = char_id
                char_id += 1
    num_char = char_id
    freq1 = np.zeros(num_char)
    for ele in dict_char:
        if ele != 'eps':
            if ele in res_op1:
                freq1[dict_char[ele]] = sum(res_op1[ele].values())
    freq2 = np.zeros(num_char)
    for ele in dict_char:
        if ele != 'eps':
            if ele in res_op2:
                freq2[dict_char[ele]] = sum(res_op2[ele].values())
    print freq1 * 1. / sum(freq1)
    print freq2 * 1. / sum(freq2)
    # sum = 0
    # for i in range(num_char):
    #     if freq1[i] != 0:
    #         sum += freq2[i] * np.log(freq2[i] / freq1[i])
# generate_file_for_align(25, 0)
#generate_noisy_channel(25, 0, 50)
analyze_noisy_channel(25, 0)
