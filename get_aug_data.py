from levenshtein import align_pair, align
from os.path import join as pjoin
from multiprocessing import Pool
import numpy as np


folder_data = '/scratch/dong.r/Dataset/OCR/data/char_25/train_test_dev/train/'


def error_rate():
    with open(pjoin(folder_data, 'train.x.txt'), 'r') as f_:
        list_x = f_.readlines()
    with open(pjoin(folder_data, 'train.y.txt'), 'r') as f_:
        list_y = f_.readlines()
    list_dis = []
    for i in range(len(list_x)):
        print i
        list_dis.append(align(list_x[i], list_y[i]))
    nthread = 50
    P = Pool(nthread)
    list_dis2 = align_pair(P, list_x, list_y, nthread, flag_char=1)
    value = sum(np.asarray(list_dis) - np.asarray(list_dis2))
    print(value)