import os
from os.path import join as pjoin
from levenshtein import align_pair, align_one2many, align_beam, align, count_pair
from multiprocessing import Pool
import numpy as np
import re
import sys
import string


# folder_data = '/Users/doreen/Documents/Experiment/dataset/OCR/'
folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR'


def compute_recall_thread(paras):
    thread_no, dict1, dict2 = paras
    num = len([ele for ele in dict1 if ele in dict2])
    return thread_no, num


def compute_recall(pool, list_y, list_x):
    paras = zip(np.arange(len(list_y)), list_y, list_x)
    results = pool.map(compute_recall_thread, paras)
    res = np.zeros(len(list_y))
    for tno, overlap in results:
        res[tno] = overlap
    return res


def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(dis_xy, len_y):
    micro_error = np.mean(dis_xy/len_y)
    macro_error = np.sum(dis_xy) / np.sum(len_y)
    return micro_error, macro_error


def evaluate_man(folder_name, prefix='dev', num=-1, flag_low=1):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    list_x = []
    for line in file(pjoin(cur_folder_data, prefix + '.x.txt')):
        if flag_low:
            line = line.lower()
        cur_str = line.strip()
        cur_str = cur_str.translate(None, string.punctuation)
        cur_dict = {}
        for ele in cur_str.split(' '):
            if len(ele.strip()) > 0:
                cur_dict[ele] = cur_dict.get(ele, 0) + 1
        list_x.append(cur_dict)
    list_x = list_x[:num]
    list_y = []
    for line in file(pjoin(cur_folder_data, prefix + '.y.txt')):
        if flag_low:
            line = line.lower()
        cur_str = line.strip()
        cur_str = cur_str.translate(None, string.punctuation)
        cur_dict = {}
        for ele in cur_str.split(' '):
            if len(ele.strip()) > 0:
                cur_dict[ele] = cur_dict.get(ele, 0) + 1
        list_y.append(cur_dict)
    list_y = list_y[:num]
    len_y = [len(y) for y in list_y]
    pool = Pool(100)
    recall_xy = compute_recall(pool, list_y, list_x)
    np.savetxt(pjoin(cur_folder_data, prefix + '.re.txt'), np.asarray(zip(recall_xy, len_y)), fmt='%d')


def evaluate_man_wit(folder_name, prefix='dev', flag_low=1):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    list_x = []
    for line in file(pjoin(cur_folder_data, prefix + '.x.txt')):
        if flag_low:
            line = line.lower()
        items = line.strip().split('\t')
        cur_dict = {}
        for cur_str in items:
            cur_str = cur_str.translate(None, string.punctuation)
            for ele in cur_str.split(' '):
                if len(ele.strip()) > 0:
                    cur_dict[ele] = cur_dict.get(ele, 0) + 1
        list_x.append(cur_dict)
    list_y = []
    for line in file(pjoin(cur_folder_data, prefix + '.y.txt')):
        if flag_low:
            line = line.lower()
        cur_str = line.strip()
        cur_str = cur_str.translate(None, string.punctuation)
        cur_dict = {}
        for ele in cur_str.split(' '):
            if len(ele.strip()) > 0:
                cur_dict[ele] = cur_dict.get(ele, 0) + 1
        list_y.append(cur_dict)
    pool = Pool(100)
    recall_xy = compute_recall(pool, list_y, list_x)
    len_y = [len(y) for y in list_y]
    np.savetxt(pjoin(cur_folder_data, prefix + '.re.txt'), np.asarray(zip(recall_xy, len_y)), fmt='%d')


cur_folder = sys.argv[1]
cur_prefix = sys.argv[2]
#evaluate_man(cur_folder, cur_prefix, -1)
evaluate_man_wit(cur_folder, cur_prefix)
