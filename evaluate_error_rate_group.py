import os
from os.path import join as pjoin
from levenshtein import align_pair, align_one2many, align_beam, align
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



def evaluate_best(folder_name, out_folder, prefix='dev', beam_size=100, start=0, end=-1):
    global folder_data
    folder_data = pjoin(folder_data, folder_name)
    if end == -1:
        file_name = pjoin(folder_data, out_folder, prefix + '.om2.txt')
    else:
        file_name = pjoin(folder_data, out_folder, prefix + '.om2.txt.' + str(start) + '_' + str(end))
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
    with open(pjoin(folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip().lower() for ele in f_.readlines()][start:end]
    len_yc = [len(y) for y in list_y]
    # len_yw = [len(y.split()) for y in list_y]
    print len(len_yc)
    nthread=100
    P = Pool(nthread)
    dis_by, best_str = align_beam(P, list_y, list_dec, 1)
    dis_ty = align_pair(P, list_y,  list_top, 1)
    # dis_xy = align_pair(P, list_y, list_x,  1)

    # dis_by_w, best_str_w = align_beam(P, list_y, list_dec, 0)
    # dis_ty_w = align_pair(P, list_y, list_top, 0)
    # dis_xy_w = align_pair(P, list_y, list_top, 0)
    dis_char = np.asarray(zip(dis_by, dis_ty, len_yc))
    # dis_word = np.asarray(zip(dis_by_w, dis_ty_w, dis_xy_w, len_yw))
    if end == -1:
        outfile_char = pjoin(folder_data, out_folder, prefix + '.ec2.txt')

        # outfile_word  = pjoin(folder_data, out_folder, prefix + '.ew1.txt')
    else:
        outfile_char = pjoin(folder_data, out_folder, prefix + '.ec2.txt.' + str(start) + '_' + str(end))
        # outfile_word = pjoin(folder_data, out_folder, prefix + '.ew1.txt.' + str(start) + '_' + str(end))
    np.savetxt(outfile_char, dis_char, fmt='%d')
    # np.savetxt(outfile_word, dis_word, fmt='%d')
    # with open(pjoin(folder_data, out_folder, prefix + '.bc.txt.' + str(start) + '_' + str(end)), 'w') as f_:
    #     for cur_str in best_str:
    #         f_.write(cur_str + '\n')
    # with open(pjoin(folder_data, out_folder, prefix + '.bw.txt.' + str(start) + '_' + str(end)), 'w') as f_:
    #     for cur_str in best_str_w:
    #         f_.write(cur_str + '\n')


def evaluate_group_ocr(P, folder_name, prefix='dev', start=0, end=-1):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    with open(pjoin(cur_folder_data, prefix + '.x.txt'), 'r') as f_:
        list_x = [ele.lower().strip('\n').split('\t')[0] for ele in f_.readlines()]
    with open(pjoin(cur_folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip().lower() for ele in f_.readlines()]
    len_yc = [len(y) for y in list_y]
    # nthread=100
    # P = Pool(nthread)
    dis_xy = align_pair(P, list_x, list_y)
    np.savetxt(pjoin(cur_folder_data, prefix + '.ec.txt.' + str(start) + '_' + str(end)), np.asarray(zip(dis_xy, len_yc)), fmt='%d')


def evaluate_all(folder_name, g1_file, g2_file, g3_file, ocr_file):
    global folder_data
    cur_folder_data = pjoin(folder_data, folder_name)
    dis1 = np.loadtxt(pjoin(cur_folder_data, g1_file), dtype=int)
    dis2 = np.loadtxt(pjoin(cur_folder_data, g2_file), dtype=int)
    dis3 = []
    for line in file(pjoin(cur_folder_data, g3_file)):
        items = line.strip('\n').split('\t')
        dis3.append([int(items[0]), int(items[1]), int(items[2]), int(items[-1])])
    dis3 = np.asarray(dis3)
    dis4 = np.loadtxt(pjoin(cur_folder_data, ocr_file), dtype=int)
    for i in range(dis1.shape[1] -1):
        micro, macro = error_rate(dis1[:,i], dis1[:, -1])
        print micro, macro
    for i in range(dis2.shape[1] -1):
        micro, macro = error_rate(dis2[:,i], dis2[:, -1])
        print micro, macro
    for i in range(dis3.shape[1] - 1):
        micro, macro = error_rate(dis3[:,i], dis3[:, -1])
        print micro, macro
    #micro, macro = error_rate(dis3, dis2[:, -1])
    #print micro, macro
    micro, macro = error_rate(dis4[:, 0], dis4[:, -1])
    print micro, macro


# if False:
#     filename = sys.argv[1]
#     error_rate_file(filename)
# else:
#     cur_folder = sys.argv[1]
#     cur_prefix = sys.argv[2]
#     cur_out = sys.argv[3]
#     beam = int(sys.argv[4])
#     start_line = int(sys.argv[5])
#     end_line = int(sys.argv[6])
#     evaluate_best(cur_folder, cur_out, cur_prefix,  beam_size=beam, start=start_line, end=end_line)
#
# pool = Pool(100)
# for line in file('output3'):
#     items = line.strip().split(' ')
#     start = int(items[0])
#     end = int(items[1])
#     evaluate_group_ocr(pool, 'richmond/0/0/50/train_new', 'group', start, end)

evaluate_all('richmond/0/0/50/train_new', 'group.ec1.txt', 'group.ec2.txt', 'group.em3.txt', 'group.ec.txt')
