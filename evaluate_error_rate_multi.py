import os
from os.path import join as pjoin
from levenshtein import align_pair, align_one2many, align_beam, align, count_pair
from multiprocessing import Pool
import numpy as np
import re
import sys
import string


# folder_data = '/Users/doreen/Documents/Experiment/dataset/OCR/'
folder_data = '/scratch/dong.r/Dataset/OCR'


def remove(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def error_rate(dis_xy, len_y):
    micro_error = np.mean(dis_xy/len_y)
    macro_error = np.sum(dis_xy) / np.sum(len_y)
    return micro_error, macro_error


def evaluate_multi(folder_name, out_folder, prefix='dev', beam_size=100, start=0, end=-1, flag_char=1, flag_low=1):
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
        cur_str = line.strip()
        # cur_str = line.strip().lower()
        #cur_str = cur_str.translate(None, string.punctuation)
        # cur_str = ' '.join([ele for ele in cur_str.split(' ') if len(ele) > 0])
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
        # list_y_old = [ele.strip().lower() for ele in f_.readlines()][start:end]
        list_y_old = [ele.strip() for ele in f_.readlines()][start:end]
        list_y = []
        for ele in list_y_old:
            #ele = ele.translate(None, string.punctuation)
            ele = ' '.join([item for item in ele.split(' ') if len(item) > 0])
            list_y.append(ele)
    if flag_char:
        len_y = [len(y) for y in list_y]
    else:
        len_y = [len(y.split()) for y in list_y]
    print len(len_y)
    nthread = 100
    pool = Pool(nthread)
    dis_by, best_str = align_beam(pool, list_y, list_dec, flag_char, flag_low)
    # num_ins, num_del, num_rep = count_pair(pool, list_top, list_y)
    dis_ty = align_pair(pool, list_y,  list_top, flag_char, flag_low)
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
    np.savetxt(outfile, dis, fmt='%d')
    if flag_char:
        outfile = pjoin(folder_data, out_folder, prefix + '.topc.txt.' + str(start) + '_' + str(end))
    else:
        outfile = pjoin(folder_data, out_folder, prefix + '.topw.txt.' + str(start) + '_' + str(end))
    with open(outfile, 'w') as f_:
        for cur_str in list_top:
            f_.write(cur_str + '\n')
    if flag_char:
        outfile = pjoin(folder_data, out_folder, prefix + '.bsc.txt.' + str(start) + '_' + str(end))
    else:
        outfile = pjoin(folder_data, out_folder, prefix + '.bsw.txt.' + str(start) + '_' + str(end))
    with open(outfile, 'w') as f_:
        for cur_str in best_str:
            f_.write(cur_str + '\n')


cur_folder = sys.argv[1]
cur_out = sys.argv[2]
cur_prefix = sys.argv[3]
beam = int(sys.argv[4])
start_line = int(sys.argv[5])
end_line = int(sys.argv[6])
flag_char = int(sys.argv[7])
evaluate_multi(cur_folder, cur_out, cur_prefix, beam_size=beam, start=start_line, end=end_line, flag_char=flag_char)
