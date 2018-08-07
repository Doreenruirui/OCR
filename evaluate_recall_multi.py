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


def evaluate_recall(folder_name, out_folder, prefix='dev', beam_size=100, start=0, end=-1, flag_low=1):
    global folder_data
    folder_data = pjoin(folder_data, folder_name)
    if end == -1:
        file_name = pjoin(folder_data, out_folder, prefix + '.o.txt')
    else:
        file_name = pjoin(folder_data, out_folder, prefix + '.o.txt.' + str(start) + '_' + str(end))
    line_id = 0
    list_dec = []
    dict_beam = {}
    list_top = []
    for line in file(file_name):
        line_id += 1
        if flag_low:
            line = line.lower()
        cur_str = line.strip()
        cur_str = cur_str.translate(None, string.punctuation)
        cur_dict = {}
        for ele in cur_str.split(' '):
            if len(ele.strip()) > 0:
                cur_dict[ele] = cur_dict.get(ele, 0) + 1
        if line_id % beam_size == 1:
            if line_id > 1:
                list_dec.append(dict_beam)
                dict_beam = {}
            list_top.append(cur_dict)
        for ele in cur_dict:
            dict_beam[ele] = dict_beam.get(ele, 0) + 1
    list_dec.append(dict_beam)
    
    if end == -1:
        end = len(list_dec)
    with open(pjoin(folder_data, prefix + '.y.txt'), 'r') as f_:
        # list_y_old = [ele.strip().lower() for ele in f_.readlines()][start:end]
        list_y_old = [ele.strip() for ele in f_.readlines()][start:end]
        list_y = []
        for line in list_y_old:
            if flag_low:
                line = line.lower()
            cur_str = line.translate(None, string.punctuation)
            dict_y = {}
            for item in cur_str.split(' '):
                if len(item.strip()) > 0:
                    dict_y[item] = dict_y.get(item, 0) + 1
            list_y.append(dict_y)
    nthread = 100
    pool = Pool(nthread)
    recall_by = compute_recall(pool, list_y, list_dec)
    recall_ty = compute_recall(pool, list_y,  list_top)
    len_y = [len(ele) for ele in list_y]
    dis = np.asarray(zip(recall_by, recall_ty, len_y))
    if end == -1:
        outfile = pjoin(folder_data, out_folder, prefix + '.re.txt')
    else:
        outfile = pjoin(folder_data, out_folder, prefix + '.re.txt.' + str(start) + '_' + str(end))
    np.savetxt(outfile, dis, fmt='%d')


cur_folder = sys.argv[1]
cur_out = sys.argv[2]
cur_prefix = sys.argv[3]
beam = int(sys.argv[4])
start_line = int(sys.argv[5])
end_line = int(sys.argv[6])
arg_flag = int(sys.argv[7])
evaluate_recall(cur_folder, cur_out, cur_prefix, beam_size=beam, start=start_line, end=end_line, flag_low=arg_flag)
