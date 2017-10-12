import os
from os.path import join as pjoin
from levenshtein import align_pair, align_one2many, align_beam, align, count_pair
from multiprocessing import Pool
import numpy as np
import re
import sys

# folder_data = '/Users/doreen/Documents/Experiment/dataset/OCR/'
folder_data = '/scratch/dong.r/Dataset/OCR'


def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(dis_xy, len_y):
    micro_error = np.mean(dis_xy/len_y)
    macro_error = np.sum(dis_xy) / np.sum(len_y)
    return micro_error, macro_error


def evaluate_multi(folder_name, out_folder, prefix='dev', beam_size=100, start=0, end=-1, flag_char=1):
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
        cur_str = line.strip().lower()
        if line_id % beam_size == 1:
            if len(list_beam) == beam_size:
                list_dec.append(list_beam)
                list_beam = []
            list_top.append(cur_str)
        list_beam.append(cur_str)
    list_dec.append(list_beam)
    if end == -1:
        end = len(list_dec)
    with open(pjoin(folder_data, prefix + '.y.txt'), 'r') as f_:
        list_y = [ele.strip().lower() for ele in f_.readlines()][start:end]
    if flag_char:
        len_y = [len(y) for y in list_y]
    else:
        len_y = [len(y.split()) for y in list_y]
    print len(len_y)
    nthread = 100
    pool = Pool(nthread)
    dis_by, best_str = align_beam(pool, list_y, list_dec, flag_char)
    num_ins, num_del, num_rep = count_pair(pool, list_top, best_str)
    # num_ins, num_del, num_rep = count_pair(pool, list_top, list_y)
    dis_ty = align_pair(pool, list_y,  list_top, flag_char)
    dis = np.asarray(zip(dis_by, dis_ty, len_y))
    if end == -1:
        if flag_char:
            outfile = pjoin(folder_data, out_folder, prefix + '.ec.txt')
        else:
            outfile = pjoin(folder_data, out_folder, prefix + '.ew.txt')
    else:
        if flag_char:
            outfile = pjoin(folder_data, out_folder, prefix + '.ec.txt.' + str(start) + '_' + str(end))
        else:
            outfile = pjoin(folder_data, out_folder, prefix + '.ew.txt.' + str(start) + '_' + str(end))
    with open(pjoin(folder_data, out_folder, prefix + 'op.txt.' + str(start) + '_' + str(end)), 'w') as f_:
        f_.write('%d\t%d\t%d\n' % (num_ins, num_del, num_rep))
    np.savetxt(outfile, dis, fmt='%d')



cur_folder = sys.argv[1]
cur_out = sys.argv[2]
cur_prefix = sys.argv[3]
beam = int(sys.argv[4])
start_line = int(sys.argv[5])
end_line = int(sys.argv[6])
evaluate_multi(cur_folder, cur_out, cur_prefix, beam_size=beam, start=start_line, end=end_line, flag_char=1)

