import os
from os.path import join as pjoin
from levenshtein import align_pair, align_one2many, align_beam, align
from multiprocessing import Pool
import numpy as np
import re
import sys

# folder_data = '/Users/doreen/Documents/Experiment/dataset/OCR/'
folder_data = '/scratch/dong.r/Dataset/OCR'


def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(dis_xy, len_y):
    micro_error = np.mean(dis_xy/len_y)
    macro_error = np.sum(dis_xy) / np.sum(len_y)
    return micro_error, macro_error

def evaluate_man(folder_name, prefix='dev', num=-1):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    with open(pjoin(cur_folder_data, prefix + '.x.txt'), 'r') as f_:
        list_x = [ele.lower().strip() for ele in f_.readlines()][:num]
    with open(pjoin(cur_folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip().lower() for ele in f_.readlines()][:num]
    len_yc = [len(y) for y in list_y]
    pool = Pool(100)
    dis_xy = align_pair(pool, list_x, list_y)
    np.savetxt(pjoin(cur_folder_data, prefix + '.ec.txt'), np.asarray(zip(dis_xy, len_yc)), fmt='%d')


def evaluate_man_wit(folder_name, prefix='dev'):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    with open(pjoin(cur_folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [remove(ele).strip().lower() for ele in f_.readlines()]
    list_x = []
    num = []
    list_y_new = []
    with open(pjoin(cur_folder_data, prefix + '.x.txt'), 'r') as f_:
        line_id = 0
        for line in f_.readlines():
            cur_line = remove(line).lower().strip('\n').split('\t')
            cur_line = [ele.strip() for ele in cur_line if len(ele.strip()) > 0]
            list_x += cur_line
            num.append(len(cur_line))
            list_y_new += [list_y[line_id] for _ in cur_line]
            line_id += 1
    pool = Pool(100)
    dis_xy = align_pair(pool, list_x, list_y_new)
    line_id = 0
    with open(pjoin(cur_folder_data, prefix + '.ec.txt'), 'w') as f_:
        for i in range(len(list_y)):
            new_line_id = line_id + num[i]
            cur_dis = dis_xy[line_id: new_line_id]
            f_.write('\t'.join(map(str, cur_dis)) + '\t' + str(len(list_y[i])) + '\n')
            line_id = new_line_id

cur_folder = sys.argv[1]
cur_prefix = sys.argv[2]
#evaluate_man(cur_folder, cur_prefix, -1)
evaluate_man_wit(cur_folder, cur_prefix)
