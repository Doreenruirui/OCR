import os
from os.path import join as pjoin
from levenshtein import align_pair
from multiprocessing import Pool
import numpy as np
import re

folder_data = '/scratch/dong.r/Dataset/OCR/data/char_date_25_0/error'
#folder_data = '/scratch/dong.r/Dataset/OCR/data/char_date_50_0/train/'
#folder_data = '/scratch/dong.r/Dataset/OCR/data/char_25/train_aug_test/aug'
def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(P, nthread, flag_char, list_x, list_y, len_y):
    dis_xy = align_pair(P, list_x, list_y, nthread, flag_char=flag_char)
    micro_error_xy = 0
    len_x = len(list_x)
    for i in range(len_x):
        micro_error_xy += dis_xy[i] * 1. / (len_y[i])
    micro_error_xy = micro_error_xy * 1. / len_x
    macro_error_xy = sum(dis_xy) * 1. / sum(len_y)
    return dis_xy, micro_error_xy, macro_error_xy


def merge_file(flag_char=1, nthread=50, beam_size=100, alpha=0., beta=0.):
    #file_list = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 165098]
    # file_list = [0, 20000, 40000]
    file_list = [0, 30000, 60000, 90000, 120000, 150000, 180000, 188601]
    index = []
    list_best_char = []
    list_best = []
    list_lm = []
    for n in range(len(file_list) - 1):
        start = file_list[n]
        end = file_list[n + 1]
        file_name = pjoin(folder_data, str(beam_size), 'dev.o.txt.' + str(start) + '_' + str(end))
        with open(file_name, 'r') as f_:
            lines = f_.readlines()
            line_id = 0
            for line in lines:
                # old = line
                line = line.split('\t')
                if len(line) < 3:
                    line_id += 1
                    continue
                # line[2] = line[2].strip()
                index.append(line_id + start)
                list_lm.append(line[0])
                if flag_char:
                    list_best_char.append(line[1])
                else:
                    list_best_char.append(line[2])
                
                list_best.append(line[3].strip())
                line_id += 1

    with open(pjoin(folder_data, 'dev.x.txt'), 'r') as f_:
        list_x = [ele.strip() for ele in f_.readlines()]
        list_x = [list_x[i] for i in index]
    with open(pjoin(folder_data, 'dev.y.txt'), 'r') as f_:
        list_y = [ele.strip() for ele in f_.readlines()]
        list_y = [list_y[i] for i in index]
    if flag_char:
        len_y = [len(y) for y in list_y]
    else:
        len_y = [len(y.split()) for y in list_y]
    print len(len_y)
    P = Pool(nthread)
    dis_ly, micro_ly, macro_ly = error_rate(P, nthread, flag_char, list_lm, list_y, len_y)
    dis_xy, micro_xy, macro_xy = error_rate(P, nthread, flag_char, list_x, list_y, len_y)
    dis_cy, micro_cy, macro_cy = error_rate(P, nthread, flag_char, list_best_char, list_y, len_y)
    dis_by, micro_by, macro_by = error_rate(P, nthread, flag_char, list_best, list_y, len_y)
    print micro_xy, micro_cy, micro_by, micro_ly
    print macro_xy, macro_cy, macro_by, macro_ly


def eval_error(flag_char=1, nthread=50, beam_size=0):
    list_o = []
    line_id = 0
    for line in file(pjoin(folder_data, '4','dev.o.txt')):
        if line_id % beam_size == 0:
            list_o.append(remove(line.strip()))
        line_id += 1
    print(len(list_o))
    with open(pjoin(folder_data, 'dev.x.txt'), 'r') as f_:
        list_x = [remove(ele.strip()) for ele in f_.readlines()]
    with open(pjoin(folder_data, 'dev.y.txt'), 'r') as f_:
        list_y = [remove(ele.strip()) for ele in f_.readlines()]
    if flag_char:
        len_y = [len(y) for y in list_y]
    else:
        len_y = [len(y.split()) for y in list_y]
    print len(len_y)
    P = Pool(nthread)
    dis_ly, micro_ly, macro_ly = error_rate(P, nthread, flag_char, list_o, list_x, len_y)
    dis_xy, micro_xy, macro_xy = error_rate(P, nthread, flag_char, list_y, list_x, len_y)
    print micro_ly, micro_ly
    #print macro_xy, macro_ly
    print micro_xy, macro_xy

#merge_file(flag_char=1, beam_size=100, alpha=0.0, beta=0.3)
eval_error(flag_char=1, nthread=40, beam_size=1)




