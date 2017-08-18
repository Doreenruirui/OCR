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


folder_multi = '/scratch/dong.r/Dataset/OCR/multi'
folder_lm = '/scratch/dong.r/Dataset/OCR/lm/char'


def rank_sent(pool, sents):
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


def get_train_data(train_id, split_id, error_ratio, lm_prob, ocr_prob, train, lm_name):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    list_x = []
    for line in file(join(folder_train, train + '.x.txt')):
        list_x.append(line)
    list_y = []
    for line in file(join(folder_train, train + '.y.txt')):
        list_y.append(line)
    folder_out = join(folder_train, 'noisy_' + str(error_ratio) + '_' + str(lm_prob) + '_' + str(ocr_prob))
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    f_x = open(join(folder_out, train + '.x.txt'), 'w')
    f_y = open(join(folder_out, train + '.y.txt'), 'w')
    f_z = open(join(folder_out, train + '.z.txt'), 'w')
    pool = Pool(100, initializer=initialize(pjoin(folder_lm, lm_name)))
    for i in range(len(list_x)):
        print i
        cur_x = [ele.strip() for ele in list_x[i].strip('\n').split('\t') if len(ele.strip()) > 0]
        best_str, best_id, best_prob, probs = rank_sent(pool, cur_x)
        if - best_prob / len(cur_x[best_id]) <= lm_prob * 0.01:
            # if best_prob / probs[0] <= error_ratio * 0.01 and - probs[0] / len(cur_x[0]) < 0.01 * ocr_prob:
            if best_prob / probs[0] <= error_ratio * 0.01:
                if align(best_str, cur_x[0]) * 1. / len(best_str) <= 0.01 * ocr_prob:
                    f_x.write(cur_x[0] + '\n')
                    f_y.write(best_str + '\n')
                    f_z.write(list_y[i])
    f_x.close()
    f_y.close()
    f_z.close()
        # nsample = len(cur_x)
        # good_index = []
        # for j in range(nsample):
        #     if - probs[j] / len(cur_x[j]) <= lm_prob * 0.01:
        #         good_index.append(j)
        # if len(good_index) > 0:
        #     bad_index = []
        #     for j in range(nsample):
        #         if j not in good_index:
        #             bad_index.append(j)
        #     for j in bad_index:
        #         for k in good_index:
        #             if probs[k] / probs[j] <= error_ratio * 0.01 and - probs[j] / len(cur_x[j]) < 2:
        #                 f_x.write(cur_x[j] + '\n')
        #                 f_y.write(cur_x[k] + '\n')
    # f_x.close()
    # f_y.close()


arg_train_id = int(sys.argv[1])
arg_split_id = int(sys.argv[2])
arg_error_ratio = int(sys.argv[3])
arg_lm_prob = int(sys.argv[4])
arg_ocr_prob = int(sys.argv[5])
arg_train = sys.argv[6]
arg_lm = sys.argv[7]
get_train_data(arg_train_id, arg_split_id, arg_error_ratio, arg_lm_prob, arg_ocr_prob, arg_train, arg_lm)