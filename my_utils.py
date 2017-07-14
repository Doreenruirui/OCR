import numpy as np
from os.path import join
from multiprocessing import Pool
from levenshtein import align_pair


def error_rate(P, nthread, flag_char, list_x, list_y):
    dis_xy = align_pair(P, list_x, list_y, nthread, flag_char=flag_char)
    res = []
    for i in range(len(dis_xy)):
        res.append([dis_xy[i], len(list_x[i]), len(list_y[i])])
    return res


def split_with_ratio(num_data, ratio):
    num_train = int(np.floor(num_data * ratio))
    rand_index = np.arange(num_data)
    np.random.shuffle(rand_index)
    index_train = rand_index[:num_train]
    index_test = rand_index[num_train:]
    return index_train, index_test


def read_lines(fn, list_str, begin, end):
    line_id = 0
    if end != -1:
        for line in file(fn):
            if line_id >= begin and line_id < end:
                list_str.append(line[:-1])
            line_id += 1
    else:
        for line in file(fn):
            if line_id >= begin:
                list_str.append(line[:-1])
            line_id += 1


def error_rate_line(file_error, file_ocr, file_truth, begin, end, nthread=40, flag_char=1, flag_strip=0):
    list_ocr = []
    list_truth = []
    read_lines(file_ocr, list_ocr, begin, end)
    read_lines(file_truth, list_truth, begin, end)
    if flag_strip == 1:
        list_ocr = [ele.strip() for ele in list_ocr]
        list_truth = [ele.strip() for ele in list_truth]
    with open(file_error, 'w') as f_:
        P = Pool(nthread)
        res = error_rate(P, nthread, flag_char, list_ocr, list_truth)
        for dis, lx, ly in res:
            f_.write('%d\t%d\t%d\n' % (dis, lx, ly))




