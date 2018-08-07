from os.path import join
import os
import sys
import numpy as np
from levenshtein import count_pair, align_pair
from multiprocessing import Pool


folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR/book'


def compute_distance_single(train_id, split_id, error_ratio, train):
    folder_data = join(folder_multi, str(train_id), str(split_id), str(error_ratio))
    list_x = []
    for line in file(join(folder_data, train + '.x.txt')):
        list_x.append(line.strip())
    list_y = []
    for line in file(join(folder_data, train + '.y.txt')):
        list_y.append(line.strip(''))
    pool = Pool(100)
    dis_xy = align_pair(pool, list_y, list_x, 1)
    with open(join(folder_data, 'distance_all'), 'w') as f_:
        for i in range(len(dis_xy)):
            dis = dis_xy[i]
            f_.write(str(dis) + '\t' + str(len(list_y[i]))+ '\n')


def compute_distance(train_id, split_id, error_ratio, train):
    folder_data = join(folder_multi, str(train_id), str(split_id), str(error_ratio))
    list_x = []
    for line in file(join(folder_data, train + '.x.txt')):
        list_x.append([ele.strip() for ele in line.strip('\n').split('\t') if len(ele.strip())])
    list_y = []
    for line in file(join(folder_data, train + '.y.txt')):
        list_y.append(line.strip())
    num_sample = len(list_x)
    list_x_new = []
    list_y_new = []
    num_x = []
    for i in range(num_sample):
        num_x.append(len(list_x[i]))
        list_x_new += list_x[i]
        list_y_new += [list_y[i] for _ in list_x[i]]
    pool = Pool(100)
    dis_xy = align_pair(pool, list_y_new, list_x_new, 1)
    start = 0
    with open(join(folder_data, 'distance_all'), 'w') as f_:
        for i in range(num_sample):
            f_.write(map(str, dis_xy[start: start + num_x[i]]) + '\t' + str(len(list_y[i]))+ '\n')
            start += num_x[i]


def compute_operation_single(folder_data, train):
    # folder_train = join(folder_multi, str(train_id), str(split_id), str(error_ratio))
    list_x = []
    for line in file(join(folder_data, train + '.x.txt')):
        list_x.append(line.strip())
    list_y = []
    len_y = 0
    for line in file(join(folder_data, train + '.y.txt')):
        list_y.append(line.strip())
        len_y += len(line.strip())
    pool = Pool(100)
    num_ins, num_del, num_rep = count_pair(pool, list_y, list_x)
    print num_ins, num_del, num_rep
    print (num_ins + num_del + num_rep) * 1. / len_y
    print num_ins * 1. / len_y, num_del * 1. / len_y, num_rep * 1. / len_y


def compute_operation_multi(folder_data, train):
    # folder_train = join(folder_multi, str(train_id), str(split_id), str(error_ratio))
    list_x = []
    for line in file(join(folder_data, train + '.x.txt')):
        list_x.append([ele.strip() for ele in line.strip('\n').split('\t') if len(ele.strip())])
    list_y = []
    for line in file(join(folder_data, train + '.y.txt')):
        list_y.append(line.strip())
    num_sample = len(list_x)
    list_x_new = []
    list_y_new = []
    num_x = []
    len_y = 0
    for i in range(num_sample):
        num_x.append(len(list_x[i]))
        list_x_new += list_x[i]
        list_y_new += [list_y[i] for _ in list_x[i]]
        len_y += num_x[i] * len(list_y[i])
    pool = Pool(100)
    num_ins, num_del, num_rep = count_pair(pool, list_y_new, list_x_new)
    print (num_ins + num_del + num_rep) * 1. / len_y
    print num_ins * 1. / len_y, num_del * 1. / len_y, num_rep * 1. / len_y


def error_statistics(train_id, split_id):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    error = []
    macro_1 = 0
    macro_2 = 0
    macro_ocr_1 = 0
    macro_ocr_2 = 0
    macro_wit_1 = 0
    macro_wit_2 = 0
    for line in file(join(folder_train, 'distance_all')):
        cur_line = np.asarray(map(float, line.strip('\n').split('\t')))
        cur_error = cur_line[:-1] / cur_line[-1]
        error.append(cur_error)
        macro_1 += sum(cur_line[:-1])
        macro_2 += cur_line[-1] * (len(cur_line) - 1)
        macro_ocr_1 += cur_line[0]
        macro_ocr_2 += cur_line[-1]
        macro_wit_1 += sum(cur_line[1:-1])
        macro_wit_2 += cur_line[-1] * (len(cur_line) - 2)
    macro = macro_1 / macro_2
    macro_ocr = macro_ocr_1 / macro_ocr_2
    macro_wit = macro_wit_1 / macro_wit_2
    print macro, macro_ocr, macro_wit


def get_train_single(train_id, split_id, error_ratio, train):
    folder_train = join(folder_multi, str(train_id), str(split_id), str(error_ratio))
    str_y = ''
    line_id = 0
    num_y = []
    for line in file(join(folder_train, train + '.y.txt')):
        str_y += line.strip()
        line_id += 1
        num_y.append(len(line.strip()))
    str_y = [ele for ele in str_y]
    print len(str_y)
    print str_y[:10]
    ins_ratio = 0.0367041080885
    del_ratio = 0.0164138089303
    rep_ratio = 0.0977654722855
    error_ratio = ins_ratio + del_ratio + rep_ratio
    ins_v = ins_ratio / (ins_ratio + del_ratio + rep_ratio)
    del_v = (ins_ratio + del_ratio) / (ins_ratio + del_ratio + rep_ratio)
    num_char = len(str_y)
    num_error = int(np.floor(num_char * error_ratio))
    voc = []
    for line in file(join(folder_train, 'vocab.dat')):
        voc.append(line.strip('\n'))
    size_voc = len(voc)
    index = np.random.choice(num_char, num_error)
    for char_id in index:
        rand_v = np.random.random()
        if rand_v < ins_v:
            rand_index = np.random.choice(size_voc, 1)[0]
            str_y[char_id] += voc[rand_index]
        elif ins_v <= rand_v < del_v:
            str_y[char_id] = ''
        else:
            cur_char = str_y[char_id]
            cur_id = 0
            rand_index = np.random.choice(size_voc - 1, 1)[0]
            for char in voc:
                if cur_char != char:
                    if cur_id == rand_index:
                        str_y[char_id] = voc[cur_id]
                        break
                    cur_id += 1
    list_new_y = []
    start = 0
    with open(join(folder_train, 'syn.' + train + '.x.txt'), 'w') as f_:
        for i in range(len(num_y)):
            list_new_y.append(''.join(str_y[start: start + num_y[i]]))
            start += num_y[i]
            f_.write(list_new_y[i] + '\n')


def get_train_multi(train_id, train):
    folder_train = join(folder_multi, str(train_id))
    num_x = []
    for line in file(join(folder_train, train + '.x.txt')):
        num_x.append(len([ele.strip() for ele in line.strip('\n').split('\t') if len(ele.strip()) > 0]))
    str_y = ''
    num_y = []
    line_id = 0
    for line in file(join(folder_train, train + '.y.txt')):
        for _ in range(num_x[line_id]):
            str_y += line.strip()
            num_y.append(len(line.strip()))
        line_id += 1
    str_y = [ele for ele in str_y]
    print len(str_y)
    print str_y[:10]
    ins_ratio = 0.0425217191234
    del_ratio = 0.0286852576065
    rep_ratio = 0.0700721470811
    error_ratio = ins_ratio + del_ratio + rep_ratio
    ins_v = ins_ratio / (ins_ratio + del_ratio + rep_ratio)
    del_v = (ins_ratio + del_ratio) / (ins_ratio + del_ratio + rep_ratio)
    num_char = len(str_y)
    num_error = int(np.floor(num_char * error_ratio))
    voc = []
    for line in file(join(folder_train, 'vocab.dat')):
        voc.append(line.strip('\n'))
    size_voc = len(voc)
    index = np.random.choice(num_char, num_error)
    for char_id in index:
        rand_v = np.random.random()
        if rand_v < ins_v:
            rand_index = np.random.choice(size_voc, 1)[0]
            str_y[char_id] += voc[rand_index]
        elif ins_v <= rand_v < del_v:
            str_y[char_id] = ''
        else:
            cur_char = str_y[char_id]
            cur_id = 0
            rand_index = np.random.choice(size_voc - 1, 1)[0]
            for char in voc:
                if cur_char != char:
                    if cur_id == rand_index:
                        str_y[char_id] = voc[cur_id]
                        break
                    cur_id += 1
    list_new_y = []
    start = 0
    for i in range(len(num_y)):
        list_new_y.append(''.join(str_y[start: start + num_y[i]]))
        start += num_y[i]
    start = 0
    with open(join(folder_train, 'syn.' + train + '.x.txt'), 'w') as f_:
        for i in range(len(num_x)):
            f_.write('\t'.join(list_new_y[start: start + num_x[i]]) + '\n')
            start += num_x[i]


arg_train_id = sys.argv[1]
arg_split_id = sys.argv[2]
arg_error = sys.argv[3]
arg_train = sys.argv[4]
# compute_distance_single(arg_train_id, arg_split_id)
# error_statistics(arg_train_id, arg_split_id)
arg_folder_data = join(folder_multi, arg_train_id, arg_split_id, arg_error)
arg_folder_data = join(arg_folder_data, 'single')
compute_operation_single(arg_folder_data, 'train')
# arg_folder_data = join(folder_multi, arg_train_id, arg_split_id, arg_error)
# compute_operation_single(arg_folder_data, 'dev')
# arg_folder_data = join(folder_multi, arg_train_id)
# compute_operation_multi(arg_folder_data, 'man_wit.train')
# arg_folder_data = join(folder_multi, arg_train_id, arg_split_id)
# compute_operation_multi(arg_folder_data, 'man_wit.dev')
get_train_single(arg_train_id, arg_split_id, arg_error, 'train')
get_train_single(arg_train_id, arg_split_id, arg_error, 'dev')
# get_train_multi(arg_train_id, 'man_wit.test')