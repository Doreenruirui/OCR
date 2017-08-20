from os.path import join
import os
import sys
import numpy as np
from levenshtein import count_pair, align_pair
from multiprocessing import Pool


folder_multi = '/scratch/dong.r/Dataset/OCR/multi'


def compute_distance(train_id, split_id):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    list_x = []
    for line in file(join(folder_train, 'man_wit.train.x.txt')):
        list_x.append([ele.strip() for ele in line.strip('\n').split('\t')])
    list_y = []
    for line in file(join(folder_train, 'man_wit.train.y.txt')):
        list_y.append(line.strip('\n'))
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
    with open(join(folder_train, 'distance_all'), 'w') as f_:
        for i in range(num_sample):
            f_.write(map(str, dis_xy[start: start + num_x[i]]) + '\t' + str(len(list_y[i]))+ '\n')
            start += num_x[i]


def compute_operation(train_id, split_id):
    folder_train = join(folder_multi, str(train_id), str(split_id))
    list_x = []
    for line in file(join(folder_train, 'man_wit.train.x.txt')):
        list_x.append([ele.strip() for ele in line.strip('\n').split('\t')])
    list_y = []
    for line in file(join(folder_train, 'man_wit.train.y.txt')):
        list_y.append(line.strip('\n'))
    num_sample = len(list_x)
    list_x_new = []
    list_y_new = []
    num_x = []
    for i in range(num_sample):
        num_x.append(len(list_x[i]))
        list_x_new += list_x[i]
        list_y_new += [list_y[i] for _ in list_x[i]]
    pool = Pool(100)
    num_ins, num_del, num_rep = count_pair(pool, list_y_new, list_x_new)
    print num_ins, num_del, num_rep


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


def get_train_data(train_id, split_id, error_ratio, train):
    folder_train = join(folder_multi, str(train_id), str(split_id), str(error_ratio))
    folder_out = join(folder_train, str(error_ratio) + '_syn')
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    str_y = ''
    num_y = []
    for line in file(join(folder_train, train + '.y.txt')):
        str_y += line.strip()
        num_y.append(len(line.strip()))
    str_y = [ele for ele in str_y]
    print len(str_y)
    print str_y[:10]
    error_ratio = 0.143838207827
    num_char = len(str_y)
    num_error = int(np.floor(num_char * error_ratio))
    voc = []
    for line in file(join(folder_train, 'vocab.dat')):
        voc.append(line.strip('\n'))
    size_voc = len(voc) - 1
    index = np.random.choice(num_char, num_error)
    for char_id in index:
        cur_char = str_y[char_id]
        cur_id = 0
        rand_index = np.random.choice(size_voc, 1)
        for char in voc:
            if cur_char != char:
                if cur_id == rand_index:
                    str_y[char_id] = voc[cur_id]
                    break
                cur_id += 1
    list_new_y = []
    start = 0
    with open(join(folder_out, train + '.x.txt'), 'w') as f_:
        for i in range(len(num_y)):
            list_new_y.append(''.join(str_y[start: start + num_y[i]]))
            start += num_y[i]
            f_.write(list_new_y[i] + '\n')


arg_train_id = int(sys.argv[1])
arg_split_id = int(sys.argv[2])
# compute_distance(arg_train_id, arg_split_id)
compute_operation(arg_train_id, arg_split_id)
# error_statistics(arg_train_id, arg_split_id)
# arg_error = int(sys.argv[3])
# train = sys.argv[4]
# get_train_data(arg_train_id, arg_split_id, arg_error, train)
#