from os.path import join
import os
import sys
import numpy as np
import re


folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR/'


def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def process_y(folder_data, train):
    with open(join(folder_multi, folder_data, train + '.y.txt'), 'r') as f_:
        lines = [ele.strip() for ele in f_.readlines()]
    with open(join(folder_multi, folder_data, train + '.y.txt'), 'w') as f_:
        for line in lines:
            f_.write(line + '\n')


def get_train_single(folder_data, train, ins_r, del_r, rep_r):
    folder_train = join(folder_multi, folder_data)
    str_y = ''
    line_id = 0
    num_y = []
    voc = {}
    for line in file(join(folder_train, train + '.y.txt')):
        str_y += line.strip()
        for ele in remove_nonascii(line.strip()):
            if ele not in voc:
                voc[ele] = 1
        line_id += 1
        num_y.append(len(line.strip()))
    str_y = [ele for ele in str_y]
    voc = voc.keys()
    #voc = list(set([ele for ele in remove_nonascii(str_y)]))
    print len(str_y)
    print str_y[:10]
    ins_ratio = ins_r * 0.01
    del_ratio = del_r * 0.01
    rep_ratio = rep_r * 0.01
    error_ratio = ins_ratio + del_ratio + rep_ratio
    ins_v = ins_ratio / (ins_ratio + del_ratio + rep_ratio)
    del_v = (ins_ratio + del_ratio) / (ins_ratio + del_ratio + rep_ratio)
    num_char = len(str_y)
    num_error = int(np.floor(num_char * error_ratio))
    size_voc = len(voc)
    index = np.random.choice(num_char, num_error)
    for char_id in index:
        rand_v = np.random.random()
        if rand_v < ins_v:
            rand_index = np.random.choice(size_voc, 1)[0]
            str_y[char_id] += voc[rand_index]
        elif ins_v <= rand_v < del_v:
            str_y[char_id] = ''
        else:
            cur_char = str_y[char_id]
            cur_id = 0
            rand_index = np.random.choice(size_voc - 1, 1)[0]
            for char in voc:
                if cur_char != char:
                    if cur_id == rand_index:
                        str_y[char_id] = voc[cur_id]
                        break
                    cur_id += 1
    list_new_y = []
    start = 0
    with open(join(folder_train, train + '.x.txt'), 'w') as f_:
        for i in range(len(num_y)):
            list_new_y.append(''.join(str_y[start: start + num_y[i]]))
            start += num_y[i]
            f_.write(list_new_y[i] + '\n')


arg_folder = sys.argv[1]
arg_train = sys.argv[2]
arg_ins = int(sys.argv[3])
arg_del = int(sys.argv[4])
arg_rep = int(sys.argv[5])
process_y(arg_folder, arg_train)
get_train_single(arg_folder, arg_train, arg_ins, arg_del, arg_rep)
