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


def get_rand_train_dev(train_id):
    folder_train = join(folder_multi, str(train_id))
    dict_id = load_obj(join(folder_train, 'split_' + str(train_id)))
    train_date = dict_id['train'].keys()
    num_date = len(train_date)
    rand_index = np.arange(num_date)
    np.random.shuffle(rand_index)
    np.savetxt(join(folder_train, 'date_random'), rand_index, fmt='%d')


folder_multi = join('/gss_gpfs_scratch/dong.r/Dataset/OCR/', sys.argv[1])
arg_train_id = int(sys.argv[2])
get_rand_train_dev(arg_train_id)
