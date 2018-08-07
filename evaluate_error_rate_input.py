import os
from os.path import join as pjoin
from levenshtein import align_pair, align_one2many, align_beam, align
from multiprocessing import Pool
import numpy as np
import re
import sys
import plot_curve


# folder_data = '/Users/doreen/Documents/Experiment/dataset/OCR/'
folder_data = '/gss_gpfs_scratch/dong.r/Dataset/OCR'

def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(dis_xy, len_y):
    micro_error = np.mean(dis_xy/len_y)
    macro_error = np.sum(dis_xy) / np.sum(len_y)
    # micro_error = 0
    # len_x = len(dis_xy)
    # num_emp = 0
    # for i in range(len_x):
    #     if len_y[i] == 0:
    #         num_emp += 1
    #     else:
    #         micro_error += dis_xy[i] * 1. / (len_y[i])
    # print num_emp
    # micro_error = micro_error * 1. / (len_x - num_emp)
    # macro_error = sum(dis_xy) * 1. / sum(len_y)
    return micro_error, macro_error


def evaluate_error(cur_folder, file_ec, file_ocr, col):
    global folder_data
    folder_data = pjoin(folder_data, cur_folder)
    group = [0, 0.1, 0.2, 0.3, 0.4, 0.5, float('inf')]
    dict_error = {}
    dict_origin = {}
    for ele in group:
        dict_error[ele] = []
        dict_origin[ele] = []
    dis_ec = np.loadtxt(pjoin(folder_data, file_ec))
    line_id = 0
    for line in file(pjoin(folder_data, file_ocr)):
        items = map(float, line.strip('\n').split(' '))
        cur_error = items[0] / items[-1]
        if cur_error == 0:
            dict_error[0].append(dis_ec[line_id, col]/ dis_ec[line_id, -1])
            dict_origin[0].append(0)
        for i in range(1, 7):
            if group[i - 1] < cur_error <= group[i]:
                dict_error[group[i]].append(dis_ec[line_id, col]/ dis_ec[line_id, -1])
                dict_origin[group[i]].append(cur_error)
                break
        line_id += 1
    for ele in group:
        print ele, np.mean(dict_error[ele]), np.mean(dict_origin[ele])

evaluate_error(sys.argv[1],sys.argv[2], sys.argv[3], int(sys.argv[4]))
