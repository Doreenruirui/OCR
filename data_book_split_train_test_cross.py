from os.path import join, exists
import os
from PyLib.operate_file import load_obj, save_obj
from levenshtein import align_pair
from multiprocessing import Pool
from collections import OrderedDict
from util_my import split_with_ratio
import numpy as np
import sys
import re


def split_train_test(num_fold, split_id):
    dict_date = load_obj(join(folder_multi, 'date_info'))
    date_list = dict_date.keys()
    num_date = len(date_list)
    size_fold = int(np.floor(num_date * 1. / num_fold))
    rand_index = np.loadtxt(join(folder_multi, 'date_random'), dtype=int)
    folder_split = join(folder_multi, str(split_id))
    if not exists(folder_split):
        os.makedirs(folder_split)
    if split_id < num_fold - 1:
        start = split_id * size_fold
        end = (split_id + 1) * size_fold
    else:
        start = split_id * size_fold
        end = num_date
    index_test = rand_index[start:end]
    index_train = np.concatenate((rand_index[:start], rand_index[end:]))
    train_id = np.sort(index_train)
    test_id = np.sort(index_test)
    train_date = OrderedDict()
    for index in train_id:
        train_date[date_list[index]] = 1
    test_date = OrderedDict()
    for index in test_id:
        test_date[date_list[index]] = 1
    save_obj(join(folder_split, 'split_' + str(split_id)),
             {'train': train_date, 'test': test_date})


folder_multi = join('/gss_gpfs_scratch/dong.r/Dataset/OCR/', sys.argv[1])
arg_num_fold = int(sys.argv[2])
arg_train_id = int(sys.argv[3])
split_train_test(arg_num_fold, arg_train_id)
