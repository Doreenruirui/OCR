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


def get_all_date():
    dict_date = OrderedDict()
    for prefix in ['man', 'man_wit']:
        for line in file(join(folder_multi, prefix + '.info.txt')):
            items = line.split('\t')
            cur_date = items[0]
            if cur_date not in dict_date:
                dict_date[cur_date] = 1
    save_obj(join(folder_multi, 'date_info'), dict_date)


def get_rand_train_test():
    dict_date = load_obj(join(folder_multi, 'date_info'))
    date_list = dict_date.keys()
    num_date = len(date_list)
    rand_index = np.arange(num_date)
    np.random.shuffle(rand_index)
    np.savetxt(join(folder_multi, 'date_random'), rand_index, fmt='%d')


folder_multi = join('/gss_gpfs_scratch/dong.r/Dataset/OCR/', sys.argv[1])
get_all_date()
get_rand_train_test()

