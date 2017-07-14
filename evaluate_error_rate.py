import os
from os.path import join as pjoin
from levenshtein import align_pair, align
from multiprocessing import Pool
import numpy as np
import re
import sys


folder_data = '/scratch/dong.r/Dataset/OCR/'


def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(P, nthread, flag_char, list_x, list_y, len_y):
    dis_xy = align_pair(P, list_x, list_y, nthread, flag_char=flag_char)
    micro_error_xy = 0
    len_x = len(list_x)
    num_emp = 0
    #dis_xy = np.zeros(len_x)
    for i in range(len_x):
        #print i
        #dis_xy[i] = align(list_x[i], list_y[i])
        if len_y[i] == 0:
            num_emp += 1
        else:
            micro_error_xy += dis_xy[i] * 1. / (len_y[i])
    print num_emp 
    micro_error_xy = micro_error_xy * 1. / (len_x - num_emp)
    macro_error_xy = sum(dis_xy) * 1. / sum(len_y)
    return dis_xy, micro_error_xy, macro_error_xy


def merge_file(folder_name, out_folder, prefix='dev', flag_lm=0, nthread=50, beam_size=100, chunk_size=30000, ndata=30000):
    global folder_data
    folder_data = pjoin(folder_data, folder_name)
    def load_dec(cur_prefix, list_dec, cur_index):
        line_id = 0
        for i in range(nfile):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, ndata)
            file_name = pjoin(folder_data, out_folder, cur_prefix + '.txt.' + str(start) + '_' + str(end))
            with open(file_name, 'r') as f_:
                lines = f_.readlines()
                for line in lines:
                    if cur_prefix[-2:] == '.b':
                        list_dec.append(line.strip())
                    else:
                        list_dec.append(line.split('\t')[0])
                    if len(line.strip()) > 0:
                        cur_index.append(line_id)
                    line_id += 1
    nfile = int(np.ceil(ndata * 1./ chunk_size))
    list_top, index_top = [], []
    load_dec(prefix + '.b', list_top, index_top)
    index = set(index_top)
    list_char, index_char = [], []
    load_dec(prefix + '.c', list_char, index_char)
    index &= set(index_char)
    list_word, index_word = [], []
    load_dec(prefix + '.w', list_word, index_word)
    index &= set(index_word)
    if flag_lm:
        list_lm, index_lm = [], []
        load_dec(prefix + '.l', list_lm, index_lm)
        index &= set(index_lm)
    index = list(index)
    list_top = [list_top[i] for i in index]
    list_char = [list_char[i] for i in index]
    list_word = [list_word[i] for i in index]
    if flag_lm:
        list_lm = [list_word[i] for i in index]
    with open(pjoin(folder_data, prefix + '.x.txt'), 'r') as f_:
        list_x = [ele.strip() for ele in f_.readlines()]
        list_x = [list_x[i] for i in index]
    with open(pjoin(folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip() for ele in f_.readlines()]
        list_y = [list_y[i] for i in index]
    len_yc = [len(y) for y in list_y]
    #len_yw = [len(y.split()) for y in list_y]
    print len(len_yc)
    P = Pool(nthread)
    if flag_lm:
        dis_ly, micro_ly, macro_ly = error_rate(P, nthread, 1, list_lm, list_y, len_yc)
    else:
        dis_ly, micro_ly, macro_ly = 1, 1, 1
    dis_xy, micro_xy, macro_xy = error_rate(P, nthread, 1, list_x, list_y, len_yc)
    dis_by, micro_by, macro_by = error_rate(P, nthread, 1, list_top, list_y, len_yc)
    dis_cy, micro_cy, macro_cy = error_rate(P, nthread, 1, list_char, list_y, len_yc)
    #dis_wy, micro_wy, macro_wy = error_rate(P, nthread, 0, list_word, list_y, len_yw)
    micro_wy= 1
    macro_wy = 1
    print micro_xy, micro_cy, micro_by, micro_wy, micro_ly
    print macro_xy, macro_cy, macro_by, macro_wy, macro_ly


def eval_error(folder_name, out_folder, prefix,flag_char=1, nthread=50, beam_size=0):
    list_o = []
    line_id = 0
    for line in file(pjoin(folder_data, folder_name, out_folder, prefix + '.o.txt')):
        if line_id % beam_size == 0:
            list_o.append(remove(line.strip()))
        line_id += 1
    print(len(list_o))
    with open(pjoin(folder_data, folder_name, prefix + '.x.txt'), 'r') as f_:
        list_x = [ele.strip() for ele in f_.readlines()]
    with open(pjoin(folder_data, folder_name, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip() for ele in f_.readlines()]
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

cur_folder = sys.argv[1]
cur_prefix = sys.argv[2]
cur_out = sys.argv[3]
merge_file(cur_folder, cur_out, cur_prefix,  flag_lm=0, beam_size=100, chunk_size=int(sys.argv[5]), ndata=int(sys.argv[4]))
#eval_error(cur_folder, cur_out, cur_prefix, flag_char=1, nthread=40, beam_size=1)

