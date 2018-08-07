import os
from os.path import join as pjoin
from levenshtein import align_pair, align_one2many, align_beam, align
from multiprocessing import Pool
import numpy as np
import re
import sys

# folder_data = '/Users/doreen/Documents/Experiment/dataset/OCR/'
folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR'


def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(dis_xy, len_y):
    micro_error = np.mean(dis_xy/len_y)
    macro_error = np.sum(dis_xy) / np.sum(len_y)
    return micro_error, macro_error


def evaluate_man(folder_name, prefix='dev', flag_char=1):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    with open(pjoin(cur_folder_data, prefix + '.x.txt'), 'r') as f_:
        list_x = [ele.lower().strip() for ele in f_.readlines()]
        #list_x = [ele.strip() for ele in f_.readlines()]
    with open(pjoin(cur_folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip().lower() for ele in f_.readlines()]
        #list_y = [ele.strip() for ele in f_.readlines()]
    if flag_char:
        len_y = [len(y) for y in list_y]
    else:
        len_y = [len(y.split()) for y in list_y]
    print len(len_y)
    pool = Pool(100)
    dis_xy = align_pair(pool, list_y, list_x, flag_char=flag_char)
    if flag_char:
        outfile = pjoin(cur_folder_data, prefix + '.ec.txt')
    else:
        outfile = pjoin(cur_folder_data, prefix + '.ew.txt')
    np.savetxt(outfile, np.asarray(zip(dis_xy, len_y)), fmt='%d')


def evaluate_man_wit(folder_name, prefix='dev', flag_char=1):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    with open(pjoin(cur_folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip().lower() for ele in f_.readlines()]
    list_x = []
    num = []
    list_y_new = []
    with open(pjoin(cur_folder_data, prefix + '.x.txt'), 'r') as f_:
        line_id = 0
        for line in f_.readlines():
            cur_line = line.lower().strip('\n').split('\t')[:100]
            cur_line = [ele.strip() for ele in cur_line if len(ele.strip()) > 0]
            list_x += cur_line
            num.append(len(cur_line))
            list_y_new += [list_y[line_id] for _ in cur_line]
            line_id += 1
    pool = Pool(100)
    dis_xy = align_pair(pool, list_x, list_y_new, flag_char=flag_char)
    line_id = 0
    if flag_char:
        outfile = pjoin(cur_folder_data, prefix + '.ec.txt')
    else:
        outfile = pjoin(cur_folder_data, prefix + '.ew.txt')
    with open(outfile, 'w') as f_:
        for i in range(len(list_y)):
            new_line_id = line_id + num[i]
            cur_dis = dis_xy[line_id: new_line_id]
            if flag_char:
                cur_len = len(list_y[i])
            else:
                cur_len = len(list_y[i].split(' '))
            f_.write('\t'.join(map(str, cur_dis)) + '\t' + str(cur_len) + '\n')
            line_id = new_line_id


cur_folder = sys.argv[1]
cur_prefix = sys.argv[2]
flag_man = int(sys.argv[3])
#evaluate_man(cur_folder, 'man_wit.test.nyt_low', 1)
#evaluate_man(cur_folder, 'man_wit.test.nyt_low', 0)
if flag_man:
    evaluate_man(cur_folder, cur_prefix, flag_char=1)
    evaluate_man(cur_folder, cur_prefix, flag_char=0)
else:
    evaluate_man_wit(cur_folder, cur_prefix, flag_char=1)
    evaluate_man_wit(cur_folder, cur_prefix, flag_char=0)
