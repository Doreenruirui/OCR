from os.path import join, exists
import numpy as np
import os
import kenlm
import re
from collections import OrderedDict
import sys
from os.path import join as pjoin
from multiprocessing import Pool
from util_lm_kenlm import score_sent, initialize
from levenshtein import align


folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR/'
folder_lm = '/gss_gpfs_scratch/dong.r/Dataset/OCR/lm/char'


def rank_sent(pool, sents):
    sents = [ele.lower() for ele in sents]
    probs = np.ones(len(sents)) * -1
    results = pool.map(score_sent, zip(np.arange(len(sents)), sents))
    max_str = ''
    max_prob = float('-inf')
    max_id = -1
    for tid, score in results:
        cur_prob = score
        probs[tid] = cur_prob
        if cur_prob > max_prob:
            max_prob = cur_prob
            max_str = sents[tid]
            max_id = tid
    return max_str, max_id, max_prob, probs


def get_train_data(cur_folder, lm_prob, train, lm_name, start, end):
    folder_train = join(folder_multi, cur_folder)
    list_info = []
    line_id = 0
    for line in file(join(folder_train, train + '.info.txt')):
        if line_id >= start:
            list_info.append(line)
            if line_id +1 == end:
                break
        line_id += 1
    list_x = []
    line_id = 0
    for line in file(join(folder_train, train + '.x.txt')):
        if line_id >= start:
            list_x.append(line)
            if line_id + 1 == end:
                break
        line_id += 1
    if 'man' in arg_train:
        list_y = []
        line_id = 0
        for line in file(join(folder_train, train + '.y.txt')):
            if line_id >= start:
                list_y.append(line)
                if line_id +1 == end:
                    break
            line_id += 1
    folder_out = join(folder_train, 'noisy_' + str(lm_prob))
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    f_x = open(join(folder_out, train + '.x.txt.' + str(start) + '_' + str(end)), 'w')
    f_y = open(join(folder_out, train + '.y.txt.' + str(start) + '_' + str(end)), 'w')
    f_info = open(join(folder_out, train + '.info.txt.' + str(start) + '_' + str(end)), 'w')
    if 'man' in arg_train:
        f_z = open(join(folder_out, train + '.z.txt.' + str(start) + '_' + str(end)), 'w')
    pool = Pool(100, initializer=initialize(pjoin(folder_lm, lm_name)))
    for i in range(len(list_x)):
        print i
        cur_x = [ele.strip() for ele in list_x[i].strip('\n').split('\t') if len(ele.strip()) > 0]
        best_str, best_id, best_prob, probs = rank_sent(pool, cur_x)
        if - best_prob / len(cur_x[best_id]) <= lm_prob * 0.01:
            f_x.write(cur_x[0] + '\n')
            f_y.write(cur_x[best_id] + '\n')
            f_info.write(list_info[i])
            if 'man' in arg_train:
                f_z.write(list_y[i])
    f_x.close()
    f_y.close()
    f_info.close()
    if 'man' in arg_train:
        f_z.close()


arg_folder = sys.argv[1]
arg_train = sys.argv[2]
arg_lm_prob = int(sys.argv[3])
arg_lm = sys.argv[4]
arg_start = int(sys.argv[5])
arg_end = int(sys.argv[6])
get_train_data(arg_folder, arg_lm_prob,
               arg_train, arg_lm,
               arg_start, arg_end)
