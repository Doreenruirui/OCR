import os
from os.path import join as pjoin
from levenshtein import align_pair, align_one2many, align_beam
from multiprocessing import Pool
import numpy as np
import re
import sys


folder_data = '/scratch/dong.r/Dataset/OCR/'


def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(dis_xy, len_y):
    micro_error = 0
    len_x = len(dis_xy)
    num_emp = 0
    for i in range(len_x):
        if len_y[i] == 0:
            num_emp += 1
        else:
            micro_error += dis_xy[i] * 1. / (len_y[i])
    print num_emp 
    micro_error = micro_error * 1. / (len_x - num_emp)
    macro_error = sum(dis_xy) * 1. / sum(len_y)
    return micro_error, macro_error


def error_rate_file(filename):
    dis = np.loadtxt(filename)
    for i in range(dis.shape[1] - 1):
        micro, macro = error_rate(dis[:, i], dis[:, -1])
        print micro, macro


def evaluate_best(folder_name, out_folder, prefix='dev', beam_size=100, start=0, end=-1):
    global folder_data
    folder_data = pjoin(folder_data, folder_name)
    if end == -1:
        file_name = pjoin(folder_data, out_folder, prefix + '.o.txt')
    else:
        file_name = pjoin(folder_data, out_folder, prefix + '.o.txt.' + str(start) + '_' + str(end))
    line_id = 0
    list_dec = []
    list_beam = []
    list_top = []
    for line in file(file_name):
        line_id += 1
        cur_str=line.split('\t')[0].strip().lower()
        if line_id % beam_size == 1:
            if len(list_beam) == beam_size:
                list_dec.append(list_beam)
                list_beam = []
            list_top.append(cur_str)
        list_beam.append(cur_str)
    list_dec.append(list_beam)
    with open(pjoin(folder_data, prefix + '.x.txt'), 'r') as f_:
        list_x = [ele.strip().lower().replace('-', '_') for ele in f_.readlines()][start:end]
    with open(pjoin(folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip().lower() for ele in f_.readlines()][start:end]
    len_yc = [len(y) for y in list_y]
    len_yw = [len(y.split()) for y in list_y]
    print len(len_yc)
    nthread=100
    P = Pool(nthread)
    dis_by, best_str = align_beam(P, list_y, list_dec, 1)
    dis_ty = align_pair(P, list_y,  list_top, 1)
    dis_xy = align_pair(P, list_y, list_x,  1)
    #dis_by_w, best_str_w = align_beam(P, list_y, list_dec, 0)
    #dis_ty_w = align_pair(P, list_y, list_top, 0)
    #dis_xy_w = align_pair(P, list_y, list_top, 0)
    dis_char = np.asarray(zip(dis_by, dis_ty, dis_xy, len_yc))
    #dis_word = np.asarray(zip(dis_by_w, dis_ty_w, dis_xy_w, len_yw))
    if end == -1:
        outfile_char = pjoin(folder_data, out_folder, prefix + '.ec.txt')

        #outfile_word  = pjoin(folder_data, out_folder, prefix + '.ew.txt')
    else:
        outfile_char = pjoin(folder_data, out_folder, prefix + '.ec.txt.' + str(start) + '_' + str(end))
        #outfile_word = pjoin(folder_data, out_folder, prefix + '.ew.txt.' + str(start) + '_' + str(end))
    np.savetxt(outfile_char, dis_char, fmt='%d')
    #np.savetxt(outfile_word, dis_word, fmt='%d')
    #with open(pjoin(folder_data, out_folder, prefix + '.bc1.txt.' + str(start) + '_' + str(end)), 'w') as f_:
    #    for cur_str in best_str:
    #        f_.write(cur_str + '\n')
    #with open(pjoin(folder_data, out_folder, prefix + '.bw1.txt.' + str(start) + '_' + str(end)), 'w') as f_:
    #    for cur_str in best_str_w:
    #        f_.write(cur_str + '\n')


if False:
    filename = sys.argv[1]
    error_rate_file(filename)
else:
    cur_folder = sys.argv[1]
    cur_prefix = sys.argv[2]
    cur_out = sys.argv[3]
    beam = int(sys.argv[4])
    start_line = int(sys.argv[5])
    end_line = int(sys.argv[6])
    evaluate_best(cur_folder, cur_out, cur_prefix,  beam_size=beam, start=start_line, end=end_line)

